#!/usr/bin/env python3
"""
Extract clean conversation logs from Claude Code's internal JSONL files

This tool parses the undocumented JSONL format used by Claude Code to store
conversations locally in ~/.claude/projects/ and exports them as clean,
readable markdown files.
"""

import argparse
import base64
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ClaudeConversationExtractor:
    """Extract and convert Claude Code conversations from JSONL to markdown."""

    CLAUDE_CODE_ICON = Path(__file__).resolve().parent.parent / "assets" / "claude-code-icon.png"
    CODE_HIGHLIGHT_STYLE = "native"
    METADATA_BLOCK_TYPES = {"claude_event", "file_attachment", "file_reference"}

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the extractor with Claude's directory and output location."""
        self.claude_dir = Path.home() / ".claude" / "projects"

        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Try multiple possible output directories
            possible_dirs = [
                Path.home() / "Desktop" / "Claude logs",
                Path.home() / "Documents" / "Claude logs",
                Path.home() / "Claude logs",
                Path.cwd() / "claude-logs",
            ]

            # Use the first directory we can create
            for dir_path in possible_dirs:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    # Test if we can write to it
                    test_file = dir_path / ".test"
                    test_file.touch()
                    test_file.unlink()
                    self.output_dir = dir_path
                    break
                except Exception:
                    continue
            else:
                # Fallback to current directory
                self.output_dir = Path.cwd() / "claude-logs"
                self.output_dir.mkdir(exist_ok=True)

        print(f"📁 Saving logs to: {self.output_dir}")

    def find_sessions(self, project_path: Optional[str] = None) -> List[Path]:
        """Find all JSONL session files, sorted by most recent first."""
        if project_path:
            search_dir = self.claude_dir / project_path
        else:
            search_dir = self.claude_dir

        sessions = []
        if search_dir.exists():
            for jsonl_file in search_dir.rglob("*.jsonl"):
                sessions.append(jsonl_file)
        return sorted(sessions, key=lambda x: x.stat().st_mtime, reverse=True)

    def extract_conversation(self, jsonl_path: Path, detailed: bool = False) -> List[Dict[str, Any]]:
        """Extract conversation messages from a JSONL file.
        
        Args:
            jsonl_path: Path to the JSONL file
            detailed: If True, include tool use, MCP responses, and system messages
        """
        conversation = []

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())

                        if entry.get("type") == "queue-operation":
                            continue

                        # Extract user messages
                        if entry.get("type") == "user":
                            msg = entry.get("message", entry)
                            if isinstance(msg, dict) and msg.get("role", "user") == "user":
                                content = msg.get("content", "")
                                blocks = self._normalize_content_blocks(
                                    content, include_non_text=detailed
                                )

                                if not detailed:
                                    blocks = [b for b in blocks if b.get("type") == "text"]

                                if not blocks:
                                    continue

                                has_text = any(b.get("type") == "text" for b in blocks)
                                has_tool_result = any(
                                    b.get("type") == "tool_result" for b in blocks
                                )
                                role = "tool_result" if has_tool_result and not has_text else "user"
                                text = self._blocks_to_plain_text(blocks)

                                if text and text.strip():
                                    conversation.append(
                                        {
                                            "role": role,
                                            "content": text,
                                            "content_blocks": blocks,
                                            "timestamp": entry.get("timestamp", ""),
                                            "uuid": entry.get("uuid", ""),
                                        }
                                    )

                        # Extract assistant messages
                        elif entry.get("type") == "assistant" and "message" in entry:
                            msg = entry["message"]
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                api_error_block = self._api_error_block_from_entry(
                                    entry, msg
                                )
                                if api_error_block:
                                    blocks = [api_error_block]
                                else:
                                    content = msg.get("content", [])
                                    blocks = self._normalize_content_blocks(
                                        content, include_non_text=detailed
                                    )

                                if not detailed and not api_error_block:
                                    blocks = [b for b in blocks if b.get("type") == "text"]

                                if not blocks:
                                    continue

                                text = self._blocks_to_plain_text(blocks)
                                message_id = msg.get("id", "")

                                if text and text.strip():
                                    self._append_or_merge_message(
                                        conversation,
                                        {
                                            "role": "assistant",
                                            "content": text,
                                            "content_blocks": blocks,
                                            "timestamp": entry.get("timestamp", ""),
                                            "message_id": message_id,
                                            "uuid": entry.get("uuid", ""),
                                        },
                                    )

                        # Claude Code stores todo reminders as attachment records.
                        elif detailed and entry.get("type") == "attachment":
                            blocks = self._blocks_from_attachment_entry(entry)
                            if blocks:
                                role = (
                                    "todo"
                                    if blocks[0].get("type") == "todo"
                                    else "metadata"
                                )
                                conversation.append(
                                    {
                                        "role": role,
                                        "content": self._blocks_to_plain_text(blocks),
                                        "content_blocks": blocks,
                                        "timestamp": entry.get("timestamp", ""),
                                        "uuid": entry.get("uuid", ""),
                                    }
                                )
                        
                        # Include tool use and system messages if detailed mode
                        elif detailed:
                            # Extract tool use events
                            if entry.get("type") == "tool_use":
                                tool_data = entry.get("tool", {})
                                tool_name = tool_data.get("name", "unknown")
                                tool_input = tool_data.get("input", {})
                                blocks = [
                                    {
                                        "type": "tool_use",
                                        "name": tool_name,
                                        "input": tool_input,
                                    }
                                ]
                                conversation.append(
                                    {
                                        "role": "tool_use",
                                        "content": self._blocks_to_plain_text(blocks),
                                        "content_blocks": blocks,
                                        "timestamp": entry.get("timestamp", ""),
                                    }
                                )
                            
                            # Extract tool results
                            elif entry.get("type") == "tool_result":
                                result = entry.get("result", {})
                                output = result.get("output", "") or result.get("error", "")
                                blocks = [
                                    {
                                        "type": "tool_result",
                                        "content": output,
                                        "is_error": bool(result.get("error")),
                                    }
                                ]
                                conversation.append(
                                    {
                                        "role": "tool_result",
                                        "content": self._blocks_to_plain_text(blocks),
                                        "content_blocks": blocks,
                                        "timestamp": entry.get("timestamp", ""),
                                    }
                                )
                            
                            # Extract system messages
                            elif entry.get("type") == "system" and "message" in entry:
                                msg = entry.get("message", "")
                                if msg:
                                    conversation.append(
                                        {
                                            "role": "system",
                                            "content": f"ℹ️ System: {msg}",
                                            "timestamp": entry.get("timestamp", ""),
                                        }
                                    )
                            elif entry.get("type") in {
                                "ai-title",
                                "file-history-snapshot",
                                "last-prompt",
                                "mode",
                                "system",
                            }:
                                block = self._metadata_block_from_entry(entry)
                                role = (
                                    "system"
                                    if entry.get("type") == "system"
                                    else "metadata"
                                )
                                conversation.append(
                                    {
                                        "role": role,
                                        "content": self._blocks_to_plain_text([block]),
                                        "content_blocks": [block],
                                        "timestamp": entry.get("timestamp", ""),
                                        "uuid": entry.get("uuid", ""),
                                    }
                                )

                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        # Silently skip problematic entries
                        continue

        except Exception as e:
            print(f"❌ Error reading file {jsonl_path}: {e}")

        return conversation

    def _extract_text_content(self, content, detailed: bool = False) -> str:
        """Extract text from various content formats Claude uses.
        
        Args:
            content: The content to extract from
            detailed: If True, include tool use blocks and other metadata
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Extract text from content array
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif detailed and item.get("type") == "tool_use":
                        # Include tool use details in detailed mode
                        tool_name = item.get("name", "unknown")
                        tool_input = item.get("input", {})
                        text_parts.append(f"\n🔧 Using tool: {tool_name}")
                        text_parts.append(f"Input: {json.dumps(tool_input, indent=2)}\n")
                    elif detailed and item.get("type") == "thinking":
                        thinking = item.get("thinking", "")
                        if thinking:
                            text_parts.append(f"\nThinking:\n{thinking}\n")
                    elif detailed and item.get("type") == "tool_result":
                        result_text = self._tool_result_to_text(item)
                        if result_text:
                            text_parts.append(f"\nTool result:\n{result_text}\n")
            return "\n".join(text_parts)
        else:
            return str(content)

    def _append_or_merge_message(
        self, conversation: List[Dict[str, Any]], message: Dict[str, Any]
    ) -> None:
        """Merge streamed assistant chunks from the same Claude message."""
        last = conversation[-1] if conversation else None
        message_id = message.get("message_id")

        if (
            last
            and message["role"] == "assistant"
            and last.get("role") == "assistant"
            and message_id
            and last.get("message_id") == message_id
        ):
            last.setdefault("content_blocks", []).extend(message.get("content_blocks", []))
            last["content"] = self._blocks_to_plain_text(last.get("content_blocks", []))
            if message.get("timestamp"):
                last["timestamp"] = message["timestamp"]
            return

        conversation.append(message)

    def _normalize_content_blocks(
        self, content: Any, include_non_text: bool = False
    ) -> List[Dict[str, Any]]:
        """Normalize Claude message content into renderable blocks."""
        if content is None:
            return []

        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []

        if isinstance(content, dict):
            content = [content]

        if not isinstance(content, list):
            return [{"type": "text", "text": str(content)}]

        blocks: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                if item:
                    blocks.append({"type": "text", "text": item})
                continue

            if not isinstance(item, dict):
                if include_non_text:
                    blocks.append({"type": "unknown", "value": item})
                continue

            block_type = item.get("type", "unknown")
            if block_type == "text":
                text = item.get("text", "")
                if text:
                    blocks.append({"type": "text", "text": text})
            elif include_non_text:
                blocks.append(dict(item))

        return blocks

    def _api_error_block_from_entry(
        self, entry: Dict[str, Any], message: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        text = self._tool_result_to_text({"content": message.get("content", [])}).strip()
        if not text:
            return None

        is_api_error = bool(entry.get("isApiErrorMessage")) or text.startswith(
            "API Error:"
        )
        if not is_api_error:
            return None

        status = entry.get("apiErrorStatus")
        if status in (None, ""):
            status_match = re.search(r"API Error:\s*(\d+)", text)
            status = status_match.group(1) if status_match else ""

        request_id_match = re.search(r"request id:\s*([^)]+)", text, re.IGNORECASE)
        request_id = request_id_match.group(1).strip() if request_id_match else ""
        fields = list(dict.fromkeys(re.findall(r"field:\s*'([^']+)'", text)))
        expected_format_match = re.search(
            r"Expected format:\s*(.+?)\.", text, re.IGNORECASE
        )
        gateway_match = re.search(r"inference gateway\s*\(([^)]+)\)", text)

        return {
            "type": "api_error",
            "status": status,
            "error": entry.get("error", ""),
            "summary": self._api_error_summary(text, status),
            "message": text,
            "request_id": request_id,
            "fields": fields,
            "expected_format": (
                expected_format_match.group(1).strip() if expected_format_match else ""
            ),
            "gateway": gateway_match.group(1).strip() if gateway_match else "",
        }

    def _api_error_summary(self, text: str, status: Any = "") -> str:
        validation_match = re.search(r"(\d+)\s+request validation errors?", text)
        if validation_match:
            return f"{validation_match.group(1)} request validation errors"

        summary = re.sub(r"^API Error:\s*\d+\s*", "", text).strip()
        summary = re.sub(r"\s*\(request id:\s*[^)]+\)", "", summary).strip()
        summary = re.sub(r"\.{2,}", ".", summary)
        if len(summary) <= 220:
            return summary
        return summary[:217].rstrip() + "..."

    def _blocks_from_attachment_entry(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        attachment = entry.get("attachment", {})
        if not isinstance(attachment, dict):
            return []

        attachment_type = attachment.get("type", "attachment")
        if attachment_type == "todo_reminder":
            return [{"type": "todo", "todos": attachment.get("content", [])}]
        if attachment_type == "file":
            return [self._file_attachment_block(attachment)]
        if attachment_type == "compact_file_reference":
            return [
                {
                    "type": "file_reference",
                    "filename": attachment.get("filename", ""),
                    "display_path": attachment.get("displayPath", ""),
                }
            ]

        return [
            self._claude_event_block(
                title=attachment_type.replace("_", " ").title(),
                summary=self._attachment_summary(attachment),
                details=attachment,
                event_type=f"attachment:{attachment_type}",
            )
        ]

    def _file_attachment_block(self, attachment: Dict[str, Any]) -> Dict[str, Any]:
        content = attachment.get("content", {})
        file_info = content.get("file", {}) if isinstance(content, dict) else {}
        filename = (
            attachment.get("filename")
            or file_info.get("filePath")
            or file_info.get("filename")
            or ""
        )
        text = file_info.get("content") if isinstance(file_info, dict) else ""
        if not text and isinstance(content, dict):
            text = content.get("text", "")
        return {
            "type": "file_attachment",
            "filename": filename,
            "display_path": attachment.get("displayPath", ""),
            "content": text or "",
            "num_lines": file_info.get("numLines") if isinstance(file_info, dict) else "",
            "start_line": file_info.get("startLine") if isinstance(file_info, dict) else "",
            "total_lines": (
                file_info.get("totalLines") if isinstance(file_info, dict) else ""
            ),
        }

    def _attachment_summary(self, attachment: Dict[str, Any]) -> str:
        attachment_type = attachment.get("type", "attachment")
        if attachment_type == "skill_listing":
            skill_count = attachment.get("skillCount")
            names = attachment.get("names") or []
            if skill_count:
                return f"{skill_count} skills available"
            if names:
                return f"{len(names)} skills available"
        if attachment_type == "hook_additional_context":
            return self._single_line_text(attachment.get("hookName", "Hook context"))
        if attachment_type == "auto_mode":
            return "Auto mode"
        if attachment_type == "workflow_keyword_request":
            return "Workflow keyword request"
        return attachment_type.replace("_", " ").title()

    def _metadata_block_from_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        entry_type = entry.get("type", "event")
        if entry_type == "ai-title":
            return self._claude_event_block(
                "AI Title",
                entry.get("aiTitle", ""),
                entry,
                entry_type,
            )
        if entry_type == "last-prompt":
            return self._claude_event_block(
                "Last Prompt",
                entry.get("lastPrompt", ""),
                entry,
                entry_type,
            )
        if entry_type == "mode":
            return self._claude_event_block(
                "Mode",
                entry.get("mode", ""),
                entry,
                entry_type,
            )
        if entry_type == "file-history-snapshot":
            snapshot = entry.get("snapshot", {})
            backups = snapshot.get("trackedFileBackups", {})
            count = len(backups) if isinstance(backups, dict) else 0
            noun = "backup" if count == 1 else "backups"
            return self._claude_event_block(
                "File History Snapshot",
                f"{count} tracked file {noun}",
                entry,
                entry_type,
            )
        if entry_type == "system":
            title = entry.get("subtype", "System").replace("_", " ").title()
            return self._claude_event_block(
                title,
                entry.get("content") or entry.get("message") or "",
                entry,
                entry_type,
            )
        return self._claude_event_block(
            entry_type.replace("_", " ").title(),
            self._single_line_text(entry.get("content", "")),
            entry,
            entry_type,
        )

    def _claude_event_block(
        self,
        title: str,
        summary: Any,
        details: Dict[str, Any],
        event_type: str,
    ) -> Dict[str, Any]:
        metadata = []
        for key in ("sessionId", "timestamp", "uuid", "messageId", "hookName"):
            value = details.get(key)
            if value not in (None, ""):
                metadata.append((key, self._single_line_text(value)))
        return {
            "type": "claude_event",
            "event_type": event_type,
            "title": title,
            "summary": self._single_line_text(summary),
            "metadata": metadata,
            "details": details,
        }

    def _blocks_to_plain_text(self, blocks: List[Dict[str, Any]]) -> str:
        """Build a searchable/plain-text fallback from normalized content blocks."""
        text_parts = []
        for block in blocks:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "thinking":
                thinking = block.get("thinking", "")
                if thinking:
                    text_parts.append(thinking)
            elif block_type == "tool_use":
                name = block.get("name", "unknown")
                if name == "TodoWrite":
                    text_parts.append(
                        self._todos_to_plain_text(block.get("input", {}).get("todos", []))
                    )
                else:
                    text_parts.append(f"Tool: {name}")
            elif block_type == "tool_result":
                text_parts.append(self._tool_result_to_text(block))
            elif block_type == "todo":
                text_parts.append(self._todos_to_plain_text(block.get("todos", [])))
            elif block_type == "image":
                text_parts.append("Image")
            elif block_type == "citation":
                text_parts.append(block.get("title") or block.get("url") or "Citation")
            elif block_type == "api_error":
                status = block.get("status", "")
                summary = block.get("summary") or block.get("message", "")
                label = f"API Error {status}".strip()
                text_parts.append(f"{label}: {summary}")
            elif block_type == "claude_event":
                title = block.get("title", "Claude Event")
                summary = block.get("summary", "")
                text_parts.append(f"{title}: {summary}".strip())
            elif block_type == "file_attachment":
                text_parts.append(
                    self._join_text_fragments(
                        [
                            f"File: {block.get('filename', '')}",
                            block.get("content", ""),
                        ]
                    )
                )
            elif block_type == "file_reference":
                label = block.get("display_path") or block.get("filename", "")
                text_parts.append(f"File reference: {label}")
            else:
                text_parts.append(block_type.replace("_", " ").title())

        return self._join_text_fragments([part for part in text_parts if part])

    def _join_text_fragments(self, parts: List[str]) -> str:
        """Join streamed text chunks without adding spaces inside partial words."""
        result = ""
        for part in parts:
            if not part:
                continue
            if not result:
                result = part
            elif (
                result.endswith((" ", "\n", "\t"))
                or part.startswith((" ", "\n", "\t", ".", ",", "!", "?", ":", ";", ")", "]"))
            ):
                result += part
            else:
                result += "\n\n" + part
        return result

    def _tool_result_to_text(self, block: Dict[str, Any]) -> str:
        """Extract display text from a Claude tool_result block."""
        content = block.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(part for part in parts if part)
        if content is None:
            return ""
        return str(content)

    def _html_pre_text(self, text: Any) -> str:
        """Escape text for raw HTML blocks without ending Markdown HTML parsing."""
        normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
        return html.escape(normalized, quote=True).replace("\n", "&#10;")

    def _single_line_text(self, text: Any) -> str:
        return " ".join(str(text).split())

    def _normalize_code_language(self, language: Any) -> str:
        """Normalize Markdown fence/path languages for HTML highlighters."""
        language_text = self._single_line_text(language).lower()
        if not language_text:
            return ""

        language_name = language_text.split()[0].removeprefix("language-")
        language_name = {
            "c++": "cpp",
            "cxx": "cpp",
            "cu": "cuda",
            "cuh": "cuda",
            "js": "javascript",
            "jsx": "javascript",
            "md": "markdown",
            "py": "python",
            "shell": "bash",
            "sh": "bash",
            "ts": "typescript",
            "tsx": "typescript",
            "yml": "yaml",
            "zsh": "bash",
        }.get(language_name, language_name)
        return re.sub(r"[^a-z0-9_+-]", "", language_name)

    def _pygments_language(self, language: str) -> str:
        return {
            "cuda": "cpp",
            "jsx": "javascript",
            "tsx": "typescript",
        }.get(language, language)

    def _highlighted_code_html(self, text: Any, language: str) -> str:
        if not language:
            return ""

        try:
            from pygments import highlight
            from pygments.formatters import HtmlFormatter
            from pygments.lexers import get_lexer_by_name
        except Exception:
            return ""

        try:
            lexer = get_lexer_by_name(self._pygments_language(language), stripnl=False)
            formatter = HtmlFormatter(
                nowrap=True,
                style=self.CODE_HIGHLIGHT_STYLE,
                classprefix="cc-syn-",
            )
            normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
            return highlight(normalized, lexer, formatter)
        except Exception:
            return ""

    def _syntax_highlight_style_defs(self) -> str:
        try:
            from pygments.formatters import HtmlFormatter
        except Exception:
            return ""

        try:
            formatter = HtmlFormatter(
                style=self.CODE_HIGHLIGHT_STYLE,
                classprefix="cc-syn-",
            )
            return "\n".join(
                line
                for line in formatter.get_style_defs(".cc-code-highlighted").splitlines()
                if line.startswith(".cc-code-highlighted")
            )
        except Exception:
            return ""

    def _recommended_syntax_overrides(self) -> str:
        return """
.cc-code-highlighted {
  color: #abb2bf;
}
.cc-code-highlighted .cc-syn-c,
.cc-code-highlighted .cc-syn-ch,
.cc-code-highlighted .cc-syn-cm,
.cc-code-highlighted .cc-syn-cp,
.cc-code-highlighted .cc-syn-cpf,
.cc-code-highlighted .cc-syn-c1,
.cc-code-highlighted .cc-syn-cs {
  color: #7f848e;
}
.cc-code-highlighted .cc-syn-k,
.cc-code-highlighted .cc-syn-kc,
.cc-code-highlighted .cc-syn-kd,
.cc-code-highlighted .cc-syn-kn,
.cc-code-highlighted .cc-syn-kp,
.cc-code-highlighted .cc-syn-kr,
.cc-code-highlighted .cc-syn-kt {
  color: #c678dd;
}
.cc-code-highlighted .cc-syn-o,
.cc-code-highlighted .cc-syn-ow {
  color: #56b6c2;
}
.cc-code-highlighted .cc-syn-s,
.cc-code-highlighted .cc-syn-sa,
.cc-code-highlighted .cc-syn-sb,
.cc-code-highlighted .cc-syn-sc,
.cc-code-highlighted .cc-syn-dl,
.cc-code-highlighted .cc-syn-sd,
.cc-code-highlighted .cc-syn-s2,
.cc-code-highlighted .cc-syn-se,
.cc-code-highlighted .cc-syn-sh,
.cc-code-highlighted .cc-syn-si,
.cc-code-highlighted .cc-syn-sx,
.cc-code-highlighted .cc-syn-sr,
.cc-code-highlighted .cc-syn-s1,
.cc-code-highlighted .cc-syn-ss {
  color: #98c379;
}
.cc-code-highlighted .cc-syn-m,
.cc-code-highlighted .cc-syn-mb,
.cc-code-highlighted .cc-syn-mf,
.cc-code-highlighted .cc-syn-mh,
.cc-code-highlighted .cc-syn-mi,
.cc-code-highlighted .cc-syn-il,
.cc-code-highlighted .cc-syn-mo {
  color: #d19a66;
}
.cc-code-highlighted .cc-syn-nb,
.cc-code-highlighted .cc-syn-bp,
.cc-code-highlighted .cc-syn-no {
  color: #56b6c2;
}
.cc-code-highlighted .cc-syn-nf,
.cc-code-highlighted .cc-syn-fm {
  color: #61afef;
}
.cc-code-highlighted .cc-syn-nc,
.cc-code-highlighted .cc-syn-ne,
.cc-code-highlighted .cc-syn-nn {
  color: #e5c07b;
}
.cc-code-highlighted .cc-syn-nt {
  color: #e06c75;
}
.cc-code-highlighted .cc-syn-na,
.cc-code-highlighted .cc-syn-nl {
  color: #d19a66;
}
.cc-code-highlighted .cc-syn-nv,
.cc-code-highlighted .cc-syn-vc,
.cc-code-highlighted .cc-syn-vg,
.cc-code-highlighted .cc-syn-vi,
.cc-code-highlighted .cc-syn-vm {
  color: #e06c75;
}
.cc-code-highlighted .cc-syn-gh,
.cc-code-highlighted .cc-syn-gu {
  color: #61afef;
  font-weight: 700;
}
.cc-code-highlighted .cc-syn-err {
  background-color: #e06c75;
  color: #1b1f27;
}
"""

    def _code_block_html(self, code_html: str, language: str = "") -> str:
        language_attr = (
            f' data-language="{html.escape(language, quote=True)}"' if language else ""
        )
        return (
            f'<div class="cc-code-block"{language_attr}>'
            '<button class="cc-copy-code" type="button" aria-label="Copy code" title="Copy code">Copy</button>'
            f"{code_html}</div>"
        )

    def _html_code_block(self, text: Any, language: str = "") -> str:
        language = self._normalize_code_language(language)
        highlighted = self._highlighted_code_html(text, language)
        if highlighted:
            safe_language = html.escape(language, quote=True)
            return self._code_block_html(
                f'<pre><code class="language-{safe_language} cc-code-highlighted">'
                f"{highlighted}</code></pre>",
                language,
            )

        class_attr = f' class="language-{html.escape(language, quote=True)}"' if language else ""
        return self._code_block_html(
            f"<pre><code{class_attr}>{self._html_pre_text(text)}</code></pre>",
            language,
        )

    def _io_card(self, label: str, body: str, css_class: str = "") -> str:
        class_attr = f"cc-card {css_class}".strip()
        return f'<div class="{class_attr}"><div class="cc-io">{label}</div>{body}</div>'

    def _line_count_label(self, text: Any) -> str:
        line_count = len(str(text).splitlines()) or (1 if text else 0)
        noun = "line" if line_count == 1 else "lines"
        return f"{line_count} {noun} of output"

    def _file_display_name(self, file_path: Any) -> str:
        path_text = self._single_line_text(file_path)
        return Path(path_text).name if path_text else "file"

    def _file_href(self, file_path: Any) -> str:
        path_text = self._single_line_text(file_path)
        if not path_text:
            return ""

        try:
            path = Path(path_text).expanduser()
            if path.is_absolute():
                return path.as_uri()
        except Exception:
            pass
        return path_text

    def _file_link_html(self, file_path: Any, label: str = "") -> str:
        path_text = self._single_line_text(file_path)
        display = label or self._file_display_name(path_text)
        if not path_text:
            return html.escape(display)

        href = self._file_href(path_text)
        return (
            f'<a class="cc-file-link" href="{html.escape(href, quote=True)}">'
            f"{html.escape(display)}</a>"
        )

    def _language_for_path(self, file_path: Any) -> str:
        suffix = Path(self._single_line_text(file_path)).suffix.lower()
        return {
            ".c": "c",
            ".cc": "cpp",
            ".cpp": "cpp",
            ".cu": "cuda",
            ".cuh": "cuda",
            ".go": "go",
            ".h": "c",
            ".hpp": "cpp",
            ".java": "java",
            ".js": "javascript",
            ".json": "json",
            ".jsx": "jsx",
            ".md": "markdown",
            ".py": "python",
            ".rs": "rust",
            ".sh": "bash",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".toml": "toml",
            ".yaml": "yaml",
            ".yml": "yaml",
        }.get(suffix, "")

    def _split_ide_selection_body(self, body: str) -> Tuple[str, str]:
        note = "This may or may not be related to the current task."
        selection = body.strip("\n")
        if selection.endswith(note):
            return selection[: -len(note)].rstrip(), note
        return selection, ""

    def _render_ide_selection_html(
        self, file_path: str, start_line: str, end_line: str, body: str
    ) -> str:
        code, note = self._split_ide_selection_body(body)
        language = self._language_for_path(file_path)
        title = (
            "Selected "
            + self._file_link_html(file_path)
            + f' <span class="cc-muted">(lines {html.escape(start_line)}-{html.escape(end_line)})</span>'
        )
        note_html = (
            f'<div class="cc-ide-note">{html.escape(note)}</div>' if note else ""
        )
        return (
            '<div class="cc-ide-selection">'
            f'<div class="cc-ide-title">{title}</div>'
            f"{self._html_code_block(code, language)}"
            f"{note_html}</div>"
        )

    def _ide_context_pattern(self) -> Any:
        return re.compile(
            r"<ide_selection>The user selected the lines "
            r"(?P<selection_start>\d+) to (?P<selection_end>\d+) from "
            r"(?P<selection_path>.+?):\n"
            r"(?P<selection_body>.*?)(?:</ide_selection>|$)"
            r"|<ide_(?:opened|open)_file>The user opened the file "
            r"(?P<opened_path>.+?) in the IDE\."
            r"(?P<opened_note>.*?)(?:</ide_(?:opened|open)_file>|$)",
            re.DOTALL,
        )

    def _render_ide_opened_file_html(self, file_path: str, note: str = "") -> str:
        title = (
            "Opened "
            + self._file_link_html(file_path)
            + ' <span class="cc-muted">(IDE)</span>'
        )
        note_text = self._single_line_text(note)
        note_html = (
            f'<div class="cc-ide-note">{html.escape(note_text)}</div>'
            if note_text
            else ""
        )
        return (
            '<div class="cc-ide-selection cc-ide-opened-file">'
            f'<div class="cc-ide-title">{title}</div>'
            f"{note_html}</div>"
        )

    def _render_ide_context_match_html(self, match: Any) -> str:
        if match.group("selection_path") is not None:
            return self._render_ide_selection_html(
                match.group("selection_path"),
                match.group("selection_start"),
                match.group("selection_end"),
                match.group("selection_body"),
            )

        return self._render_ide_opened_file_html(
            match.group("opened_path"),
            match.group("opened_note") or "",
        )

    def _render_text_markdown(self, text: Any) -> str:
        content = str(text)
        pattern = self._ide_context_pattern()
        rendered_parts = []
        position = 0

        for match in pattern.finditer(content):
            before = content[position : match.start()].strip()
            if before:
                rendered_parts.append(self._balance_markdown_fences(before))

            rendered_parts.append(self._render_ide_context_match_html(match))
            position = match.end()

        after = content[position:].strip()
        if after:
            rendered_parts.append(self._balance_markdown_fences(after))

        if rendered_parts:
            return "\n\n".join(rendered_parts)
        return self._balance_markdown_fences(content)

    def _render_text_html(self, text: Any) -> str:
        content = str(text)
        pattern = self._ide_context_pattern()
        rendered_parts = []
        position = 0

        for match in pattern.finditer(content):
            before = content[position : match.start()].strip()
            if before:
                rendered_parts.append(
                    self._simple_markdown_html(self._balance_markdown_fences(before))
                )

            rendered_parts.append(self._render_ide_context_match_html(match))
            position = match.end()

        after = content[position:].strip()
        if after:
            rendered_parts.append(
                self._simple_markdown_html(self._balance_markdown_fences(after))
            )

        if rendered_parts:
            return "\n\n".join(rendered_parts)
        return self._simple_markdown_html(self._balance_markdown_fences(content))

    def _render_limited_tool_output(self, result_text: str) -> str:
        lines = result_text.splitlines()
        preview = "\n".join(lines[:2])
        rest = "\n".join(lines[2:])
        output_label = self._line_count_label(result_text)
        label_html = (
            'OUT <span class="cc-muted">'
            + html.escape(output_label)
            + "</span>"
        )
        body = self._html_code_block(preview or result_text)

        if rest:
            remaining = len(lines[2:])
            remaining_label = (
                f"Show {remaining} more line"
                + ("" if remaining == 1 else "s")
            )
            body += (
                '<details class="cc-more-output">'
                f"<summary>{html.escape(remaining_label)}</summary>"
                f"{self._html_code_block(rest)}"
                "</details>"
            )

        return self._io_card(label_html, body, css_class="cc-output-card")

    def _render_limited_thinking_body(self, thinking: str) -> str:
        preview, rest, remaining = self._split_markdown_preview(thinking, max_lines=10)
        body = self._thinking_markdown_block_html(preview or thinking)

        if rest:
            remaining_label = (
                f"Show {remaining} more line"
                + ("" if remaining == 1 else "s")
            )
            body += (
                '<details class="cc-more-output cc-thinking-more">'
                f"<summary>{html.escape(remaining_label)}</summary>"
                f"{self._thinking_markdown_block_html(rest)}"
                "</details>"
            )

        return body

    def _thinking_markdown_block_html(self, thinking: str) -> str:
        rendered = self._simple_markdown_html(self._balance_markdown_fences(thinking))
        return f'<div class="cc-markdown-block cc-thinking-body">{rendered}</div>'

    def _split_markdown_preview(
        self, markdown_text: str, max_lines: int
    ) -> Tuple[str, str, int]:
        lines = markdown_text.splitlines()
        if len(lines) <= max_lines:
            return markdown_text, "", 0

        preview_lines = lines[:max_lines]
        rest_lines = lines[max_lines:]
        fence_opener = ""

        for line in preview_lines:
            stripped = line.strip()
            if not stripped.startswith("```"):
                continue
            if fence_opener:
                fence_opener = ""
            else:
                fence_opener = stripped

        if fence_opener:
            preview_lines.append("```")
            rest_lines.insert(0, fence_opener)

        return "\n".join(preview_lines), "\n".join(rest_lines), len(lines[max_lines:])

    def _read_lines_label(self, tool_input: Dict[str, Any]) -> str:
        offset = tool_input.get("offset")
        limit = tool_input.get("limit")
        start = 1

        try:
            if offset not in (None, ""):
                start = max(1, int(offset) + 1)
        except (TypeError, ValueError):
            return ""

        try:
            if limit not in (None, ""):
                limit_int = int(limit)
                if limit_int > 0:
                    return f"lines {start}-{start + limit_int - 1}"
        except (TypeError, ValueError):
            return ""

        if offset not in (None, ""):
            return f"from line {start}"
        return ""

    def _math_html(self, math_text: str, display: bool = False) -> str:
        normalized = str(math_text).replace("\r\n", "\n").replace("\r", "\n").strip()
        escaped = html.escape(normalized, quote=True)
        if display:
            return f'<div class="cc-math-block">{escaped}</div>'
        return f'<span class="cc-math-inline">{escaped}</span>'

    def _protect_math_segments(self, text: str) -> Tuple[str, Dict[str, str]]:
        placeholders: Dict[str, str] = {}
        pattern = re.compile(
            r"(\\\[(?:.|\n)*?\\\]|\\\((?:.|\n)*?\\\)|"
            r"\$\$(?:.|\n)*?\$\$|(?<!\\)\$(?!\$)(?:\\.|[^$])*?(?<!\\)\$)"
        )

        def replace(match: re.Match[str]) -> str:
            placeholder = f"CCMATHPLACEHOLDER{len(placeholders)}"
            math_text = match.group(0)
            placeholders[placeholder] = self._math_html(math_text)
            return placeholder

        return pattern.sub(replace, text), placeholders

    def _inline_markdown_html(self, text: str) -> str:
        protected_text, math_placeholders = self._protect_math_segments(text)
        escaped = html.escape(protected_text, quote=True)
        escaped = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            r'<a href="\2">\1</a>',
            escaped,
        )
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
        for placeholder, math_html in math_placeholders.items():
            escaped = escaped.replace(placeholder, math_html)
        return escaped

    def _is_table_separator(self, line: str) -> bool:
        return bool(re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", line))

    def _is_escaped(self, text: str, index: int) -> bool:
        backslashes = 0
        i = index - 1
        while i >= 0 and text[i] == "\\":
            backslashes += 1
            i -= 1
        return backslashes % 2 == 1

    def _split_markdown_table_row(self, row: str) -> List[str]:
        line = row.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|") and not self._is_escaped(line, len(line) - 1):
            line = line[:-1]

        cells = []
        cell = []
        i = 0
        in_code = False
        math_end = ""

        while i < len(line):
            if math_end:
                if math_end in {"\\)", "\\]"} and line.startswith(math_end, i):
                    cell.append(math_end)
                    i += len(math_end)
                    math_end = ""
                    continue
                if (
                    math_end == "$$"
                    and line.startswith("$$", i)
                    and not self._is_escaped(line, i)
                ):
                    cell.append("$$")
                    i += 2
                    math_end = ""
                    continue
                if (
                    math_end == "$"
                    and line[i] == "$"
                    and not self._is_escaped(line, i)
                ):
                    cell.append("$")
                    i += 1
                    math_end = ""
                    continue

                cell.append(line[i])
                i += 1
                continue

            if line.startswith("$$", i) and not self._is_escaped(line, i):
                cell.append("$$")
                i += 2
                math_end = "$$"
                continue
            if line.startswith("\\(", i):
                cell.append("\\(")
                i += 2
                math_end = "\\)"
                continue
            if line.startswith("\\[", i):
                cell.append("\\[")
                i += 2
                math_end = "\\]"
                continue

            char = line[i]
            if char == "`" and not self._is_escaped(line, i):
                in_code = not in_code
            elif char == "$" and not in_code and not self._is_escaped(line, i):
                cell.append(char)
                i += 1
                math_end = "$"
                continue
            elif char == "|" and not in_code and not self._is_escaped(line, i):
                cells.append("".join(cell).strip())
                cell.clear()
                i += 1
                continue

            cell.append(char)
            i += 1

        cells.append("".join(cell).strip())
        return cells

    def _render_markdown_table_html(self, rows: List[str]) -> str:
        parsed_rows = []
        for row in rows:
            cells = self._split_markdown_table_row(row)
            if cells:
                parsed_rows.append(cells)

        if not parsed_rows:
            return ""

        header = parsed_rows[0]
        body_rows = [
            row for row in parsed_rows[1:] if not self._is_table_separator("|".join(row))
        ]
        head_html = "".join(
            f"<th>{self._inline_markdown_html(cell)}</th>" for cell in header
        )
        body_html = "".join(
            "<tr>"
            + "".join(f"<td>{self._inline_markdown_html(cell)}</td>" for cell in row)
            + "</tr>"
            for row in body_rows
        )
        return (
            '<table class="cc-md-table"><thead><tr>'
            + head_html
            + "</tr></thead><tbody>"
            + body_html
            + "</tbody></table>"
        )

    def _simple_markdown_html(self, markdown_text: str) -> str:
        """Convert the small Markdown subset used in Agent prompts to HTML."""
        lines = markdown_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        html_parts = []
        paragraph = []
        list_type: Optional[str] = None
        code_lines: Optional[List[str]] = None
        code_language = ""
        math_lines: Optional[List[str]] = None
        math_end = ""
        table_rows: List[str] = []

        def flush_paragraph() -> None:
            if paragraph:
                html_parts.append(
                    "<p>"
                    + self._inline_markdown_html(" ".join(part.strip() for part in paragraph))
                    + "</p>"
                )
                paragraph.clear()

        def close_list() -> None:
            nonlocal list_type
            if list_type:
                html_parts.append(f"</{list_type}>")
                list_type = None

        def flush_table() -> None:
            if table_rows:
                html_parts.append(self._render_markdown_table_html(table_rows))
                table_rows.clear()

        def render_math_block(lines: List[str]) -> None:
            html_parts.append(self._math_html("\n".join(lines), display=True))

        for raw_line in lines:
            stripped = raw_line.strip()

            if stripped.startswith("```"):
                flush_paragraph()
                close_list()
                flush_table()
                if code_lines is None:
                    code_language = self._normalize_code_language(stripped[3:].strip())
                    code_lines = []
                else:
                    html_parts.append(
                        self._html_code_block("\n".join(code_lines), code_language)
                    )
                    code_lines = None
                    code_language = ""
                continue

            if code_lines is not None:
                code_lines.append(raw_line)
                continue

            if math_lines is not None:
                math_lines.append(raw_line)
                if stripped.endswith(math_end):
                    render_math_block(math_lines)
                    math_lines = None
                    math_end = ""
                continue

            display_math_start = ""
            display_math_end = ""
            if stripped.startswith("$$"):
                display_math_start = "$$"
                display_math_end = "$$"
            elif stripped.startswith("\\["):
                display_math_start = "\\["
                display_math_end = "\\]"

            if display_math_start:
                flush_paragraph()
                close_list()
                flush_table()

                rest = stripped[len(display_math_start) :]
                if rest and display_math_end in rest:
                    render_math_block([raw_line])
                else:
                    math_lines = [raw_line]
                    math_end = display_math_end
                continue

            if not stripped:
                flush_paragraph()
                close_list()
                flush_table()
                continue

            if stripped == "---":
                flush_paragraph()
                close_list()
                flush_table()
                html_parts.append("<hr>")
                continue

            if stripped.startswith("|") and "|" in stripped[1:]:
                flush_paragraph()
                close_list()
                table_rows.append(stripped)
                continue

            flush_table()

            heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading:
                flush_paragraph()
                close_list()
                level = min(6, len(heading.group(1)) + 1)
                html_parts.append(
                    f"<h{level}>{self._inline_markdown_html(heading.group(2))}</h{level}>"
                )
                continue

            ordered = re.match(r"^\d+\.\s+(.+)$", stripped)
            unordered = re.match(r"^[-*]\s+(.+)$", stripped)
            if ordered or unordered:
                flush_paragraph()
                desired_list = "ol" if ordered else "ul"
                if list_type != desired_list:
                    close_list()
                    html_parts.append(f"<{desired_list}>")
                    list_type = desired_list
                item_text = (ordered or unordered).group(1)
                html_parts.append(f"<li>{self._inline_markdown_html(item_text)}</li>")
                continue

            close_list()
            paragraph.append(stripped)

        flush_paragraph()
        close_list()
        flush_table()
        if code_lines is not None:
            html_parts.append(self._html_code_block("\n".join(code_lines), code_language))
        if math_lines is not None:
            render_math_block(math_lines)

        return "".join(html_parts)

    def _balance_markdown_fences(self, markdown_text: str) -> str:
        fence_count = 0
        for line in markdown_text.splitlines():
            if re.match(r"^\s*```", line):
                fence_count += 1
        if fence_count % 2 == 0:
            return markdown_text
        return markdown_text.rstrip() + "\n```"

    def _render_agent_input_markdown(self, tool_input: Dict[str, Any]) -> str:
        prompt = str(tool_input.get("prompt") or "")
        description = str(tool_input.get("description") or "")
        subagent_type = str(tool_input.get("subagent_type") or "")

        meta = []
        if subagent_type:
            meta.append(
                f"<span>Subagent: {html.escape(self._single_line_text(subagent_type))}</span>"
            )
        if description:
            meta.append(f"<span>{html.escape(self._single_line_text(description))}</span>")

        meta_html = (
            '<div class="cc-agent-meta">' + "".join(meta) + "</div>" if meta else ""
        )
        prompt_html = self._simple_markdown_html(prompt) if prompt else ""
        body = (
            '<div class="cc-agent-block">'
            + meta_html
            + f'<div class="cc-markdown-block">{prompt_html}</div>'
            + "</div>"
        )
        return self._io_card("IN", body, css_class="cc-agent-card")

    def _todos_to_plain_text(self, todos: Any) -> str:
        if not isinstance(todos, list):
            return ""

        lines = []
        for todo in todos:
            if not isinstance(todo, dict):
                continue
            status = todo.get("status", "pending")
            content = todo.get("content") or todo.get("activeForm") or ""
            if content:
                lines.append(f"{status}: {content}")
        return "\n".join(lines)

    def _markdown_style_block(self, icon_uri: str = "") -> str:
        icon_css = ""
        if icon_uri:
            safe_icon_uri = icon_uri.replace('"', "%22")
            icon_css = f""".cc-claude-heading {{
  display: flex;
  align-items: center;
  gap: 0.45rem;
}}
.cc-claude-heading::before {{
  content: "";
  width: 1.25rem;
  height: 1.25rem;
  display: inline-block;
  flex: 0 0 auto;
  border-radius: 50%;
  background: url("{safe_icon_uri}") center / contain no-repeat;
}}
"""

        return """<style>
.cc-log,
.cc-ide-selection,
.cc-code-block {
  --cc-muted: var(--vscode-descriptionForeground, #8a8a8a);
  --cc-fg: var(--vscode-foreground, #d4d4d4);
  --cc-line: var(--vscode-editorWidget-border, #3a3a3a);
  --cc-card: var(--vscode-editorWidget-background, #252526);
  --cc-card-2: var(--vscode-input-background, #2d2d2d);
  --cc-code-bg: #111820;
  --cc-code-bg-2: #18212b;
  --cc-focus: var(--vscode-focusBorder, #4daafc);
  --cc-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --cc-green: #73d69a;
  --cc-orange: #d97757;
  --cc-red: #d73a49;
  --cc-blue: var(--vscode-textLink-foreground, #4daafc);
  color: var(--cc-fg);
}
.cc-timeline {
  border-left: 1px solid var(--cc-line);
  margin: 0.45rem 0 1rem 0.62rem;
  padding-left: 1.15rem;
}
.cc-step {
  position: relative;
  margin: 0 0 0.9rem 0;
  min-height: 1rem;
}
.cc-step:last-child {
  margin-bottom: 0;
}
.cc-dot {
  position: absolute;
  left: -1.42rem;
  top: 0.45rem;
  width: 0.5rem;
  height: 0.5rem;
  border-radius: 50%;
  border: 2px solid var(--cc-card);
  background: var(--cc-muted);
}
.cc-step.cc-thinking .cc-dot {
  background: var(--cc-orange);
}
.cc-step.cc-api-error .cc-dot {
  background: var(--cc-red);
}
.cc-step.cc-tool .cc-dot,
.cc-step.cc-todo .cc-dot {
  background: var(--cc-green);
}
.cc-step summary {
  cursor: pointer;
  color: var(--cc-muted);
  list-style: none;
}
.cc-step summary:hover .cc-title,
.cc-step summary:focus-visible .cc-title {
  color: var(--cc-fg);
}
.cc-step summary::-webkit-details-marker {
  display: none;
}
.cc-title {
  color: var(--cc-fg);
  font-weight: 600;
}
.cc-muted {
  color: var(--cc-muted);
  font-weight: 400;
}
.cc-body {
  margin-top: 0.45rem;
}
.cc-card {
  border: 1px solid var(--cc-line);
  border-radius: 6px;
  background: var(--cc-card);
  margin-top: 0.45rem;
  padding: 0.55rem 0.65rem;
}
.cc-card code {
  font-family: var(--cc-mono);
}
.cc-card :not(pre) > code {
  white-space: pre-wrap;
}
.cc-io {
  color: var(--cc-muted);
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  margin: 0.05rem 0 0.35rem;
  text-transform: uppercase;
}
.cc-card pre,
.cc-unknown pre {
  background: var(--cc-card-2);
  border: 1px solid var(--cc-line);
  border-radius: 6px;
  font-family: var(--cc-mono);
  font-size: 0.92em;
  margin: 0;
  padding: 0.72rem 0.82rem;
  overflow-x: auto;
  white-space: pre-wrap;
}
.cc-code-block {
  background: linear-gradient(180deg, var(--cc-code-bg-2), var(--cc-code-bg));
  border: 1px solid var(--cc-line);
  border-radius: 6px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
  overflow: hidden;
  position: relative;
}
.cc-code-block[data-language]::before {
  content: attr(data-language);
  background: rgba(0, 0, 0, 0.22);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 999px;
  color: #c7d2e0;
  font: 600 0.68rem var(--cc-mono);
  letter-spacing: 0.04em;
  padding: 0.12rem 0.4rem;
  position: absolute;
  left: 0.74rem;
  text-transform: uppercase;
  top: 0.48rem;
  z-index: 1;
}
.cc-code-block pre {
  background: transparent;
  border: 0;
  border-radius: 0;
  color: var(--cc-fg);
  font-family: var(--cc-mono);
  font-size: 0.92em;
  line-height: 1.55;
  margin: 0;
  overflow: auto;
  padding: 0.82rem 4.9rem 0.82rem 0.9rem;
  white-space: pre;
}
.cc-code-block[data-language] pre {
  padding-top: 2.05rem;
}
.cc-code-block pre code {
  background: transparent;
  display: block;
  font-family: var(--cc-mono);
  min-width: max-content;
  padding: 0;
  white-space: inherit;
}
.cc-card > .cc-code-block {
  border-bottom: 0;
  border-left: 0;
  border-radius: 0 0 6px 6px;
  border-right: 0;
  margin: 0.45rem -0.65rem -0.55rem;
}
.cc-copy-code {
  background: rgba(17, 24, 32, 0.9);
  border: 1px solid var(--cc-line);
  border-radius: 4px;
  color: #c7d2e0;
  cursor: pointer;
  font: 600 0.72rem var(--cc-mono);
  line-height: 1;
  opacity: 0.72;
  padding: 0.28rem 0.42rem;
  position: absolute;
  right: 0.45rem;
  top: 0.38rem;
  transition: color 120ms ease, opacity 120ms ease, border-color 120ms ease;
  z-index: 1;
}
.cc-code-block:hover .cc-copy-code,
.cc-copy-code:focus {
  border-color: var(--cc-focus);
  color: var(--cc-fg);
  opacity: 1;
}
.cc-todo-list {
  list-style: none;
  margin: 0.55rem 0 0;
  padding: 0;
}
.cc-todo-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0.4rem 0;
}
.cc-checkbox {
  width: 0.82rem;
  height: 0.82rem;
  border: 1px solid var(--cc-muted);
  border-radius: 2px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.65rem;
}
.cc-in-progress .cc-checkbox {
  border-color: var(--cc-green);
  color: var(--cc-green);
}
.cc-collapse {
  padding: 0;
}
.cc-collapse > summary {
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.45rem;
  list-style: none;
  padding: 0.5rem 0.65rem;
}
.cc-collapse > summary::-webkit-details-marker {
  display: none;
}
.cc-collapse > summary .cc-io {
  margin: 0;
}
.cc-collapse > summary .cc-muted {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cc-collapse pre {
  margin: 0;
}
.cc-collapse .cc-code-block {
  margin: 0 0.55rem 0.55rem;
}
.cc-agent-meta {
  color: var(--cc-muted);
  display: flex;
  flex-wrap: wrap;
  font-size: 0.85rem;
  gap: 0.45rem;
  margin-bottom: 0.45rem;
}
.cc-agent-meta span {
  border: 1px solid var(--cc-line);
  border-radius: 4px;
  padding: 0.1rem 0.35rem;
}
.cc-markdown-block {
  background: var(--cc-card-2);
  border: 1px solid var(--cc-line);
  border-radius: 6px;
  padding: 0.65rem 0.75rem;
}
.cc-thinking-body {
  color: var(--cc-fg);
  font-size: 0.94em;
}
.cc-markdown-block h2,
.cc-markdown-block h3,
.cc-markdown-block h4,
.cc-markdown-block h5,
.cc-markdown-block h6 {
  color: var(--cc-fg);
  font-weight: 650;
  line-height: 1.25;
  margin: 0.2rem 0 0.6rem;
}
.cc-markdown-block h2 {
  font-size: 1.2rem;
}
.cc-markdown-block h3 {
  font-size: 1.08rem;
}
.cc-markdown-block h4,
.cc-markdown-block h5,
.cc-markdown-block h6 {
  font-size: 1rem;
}
.cc-markdown-block p,
.cc-markdown-block ol,
.cc-markdown-block ul {
  margin: 0 0 0.65rem;
}
.cc-markdown-block p:last-child,
.cc-markdown-block ol:last-child,
.cc-markdown-block ul:last-child {
  margin-bottom: 0;
}
.cc-markdown-block ol,
.cc-markdown-block ul {
  padding-left: 1.2rem;
}
.cc-markdown-block code {
  background: var(--cc-card);
  border-radius: 3px;
  font-family: var(--cc-mono);
  padding: 0.05rem 0.2rem;
}
.cc-markdown-block pre {
  font-family: var(--cc-mono);
}
.cc-markdown-block > .cc-code-block {
  border-left: 0;
  border-radius: 0;
  border-right: 0;
  margin: 0.65rem -0.75rem;
}
.cc-markdown-block > .cc-code-block:first-child {
  border-top: 0;
  border-radius: 6px 6px 0 0;
  margin-top: -0.65rem;
}
.cc-markdown-block > .cc-code-block:last-child {
  border-bottom: 0;
  border-radius: 0 0 6px 6px;
  margin-bottom: -0.65rem;
}
.cc-markdown-block pre code,
.cc-card pre code,
.cc-ide-selection pre code {
  font-family: var(--cc-mono);
  background: transparent;
  padding: 0;
}
.cc-math-block {
  display: block;
  margin: 0.75rem 0;
  overflow-x: auto;
  overflow-y: visible;
  padding: 0.18rem 0.05rem;
}
.cc-math-inline {
  overflow-y: visible;
}
.cc-math-block mjx-container,
.cc-math-inline mjx-container {
  overflow: visible !important;
  padding: 0.08em 0;
}
.cc-math-block mjx-container[jax="CHTML"] {
  max-width: none;
  min-width: max-content;
  overflow-y: visible !important;
  padding: 0.08em 0;
}
.cc-math-block mjx-container[display="true"] {
  margin: 0.25rem 0 !important;
  padding: 0.18rem 0;
}
.cc-code-highlighted {
  color: var(--cc-fg);
}
""" + (
            self._syntax_highlight_style_defs()
            + self._recommended_syntax_overrides()
        ) + """
.cc-output-card .cc-io {
  display: flex;
  gap: 0.45rem;
}
.cc-more-output {
  margin-top: 0.45rem;
}
.cc-more-output > summary {
  color: var(--cc-muted);
  cursor: pointer;
  font-size: 0.85rem;
  width: fit-content;
}
.cc-more-output > summary:hover,
.cc-more-output > summary:focus-visible {
  color: var(--cc-fg);
}
.cc-more-output pre {
  margin: 0;
}
.cc-more-output .cc-code-block {
  margin-top: 0.45rem;
}
.cc-thinking-more .cc-thinking-body {
  margin-top: 0.45rem;
}
.cc-error-card {
  background: rgba(215, 58, 73, 0.12);
  border-color: rgba(215, 58, 73, 0.45);
}
.cc-error-summary {
  color: var(--cc-fg);
  font-weight: 650;
  margin-bottom: 0.45rem;
}
.cc-error-meta,
.cc-error-fields {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}
.cc-error-meta {
  margin-bottom: 0.4rem;
}
.cc-error-meta span {
  border: 1px solid rgba(215, 58, 73, 0.35);
  border-radius: 4px;
  color: #fdaeb7;
  font-size: 0.78rem;
  padding: 0.12rem 0.38rem;
}
.cc-error-fields code {
  background: rgba(215, 58, 73, 0.16);
  border: 1px solid rgba(215, 58, 73, 0.28);
  border-radius: 4px;
  color: #ffdce0;
  font-family: var(--cc-mono);
  padding: 0.08rem 0.28rem;
}
.cc-event-card {
  background: rgba(77, 170, 252, 0.08);
  border-color: rgba(77, 170, 252, 0.3);
}
.cc-event-summary {
  color: var(--cc-fg);
  font-weight: 600;
  margin-bottom: 0.4rem;
}
.cc-event-summary:last-child {
  margin-bottom: 0;
}
.cc-event-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}
.cc-event-meta span {
  border: 1px solid rgba(77, 170, 252, 0.24);
  border-radius: 4px;
  color: #9fd2ff;
  font-size: 0.78rem;
  padding: 0.12rem 0.38rem;
}
.cc-file-link {
  color: var(--cc-blue);
  text-decoration: none;
}
.cc-file-link:hover {
  text-decoration: underline;
}
.cc-read-summary {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cc-md-table {
  border-collapse: collapse;
  display: block;
  margin: 0 0 0.75rem;
  overflow-x: auto;
  width: 100%;
}
.cc-md-table th,
.cc-md-table td {
  border: 1px solid var(--cc-line);
  padding: 0.3rem 0.45rem;
  text-align: left;
  vertical-align: top;
}
.cc-md-table th {
  background: var(--cc-card);
  font-weight: 600;
}
.cc-ide-selection {
  border: 1px solid var(--cc-line);
  border-radius: 6px;
  background: var(--cc-card);
  margin: 0.65rem 0;
  padding: 0.6rem 0.7rem;
}
.cc-ide-title {
  font-weight: 600;
  margin-bottom: 0.45rem;
}
.cc-ide-opened-file .cc-ide-title {
  margin-bottom: 0;
}
.cc-ide-selection pre {
  background: var(--cc-card-2);
  border: 1px solid var(--cc-line);
  border-radius: 6px;
  font-family: var(--cc-mono);
  font-size: 0.92em;
  margin: 0;
  overflow-x: auto;
  padding: 0.65rem 0.75rem;
  white-space: pre-wrap;
}
.cc-ide-note {
  color: var(--cc-muted);
  font-size: 0.85rem;
  margin-top: 0.45rem;
}
""" + icon_css + """</style>
"""

    def _render_message_markdown(self, msg: Dict[str, Any]) -> str:
        """Render a conversation message using VS Code-friendly Markdown."""
        blocks = msg.get("content_blocks")
        if not blocks:
            return self._render_text_markdown(msg.get("content", ""))

        role = msg.get("role", "")
        if role == "assistant":
            return self._render_assistant_turn_markdown([msg])

        text_only = all(block.get("type") == "text" for block in blocks)
        if text_only:
            content = self._join_text_fragments(
                [block.get("text", "") for block in blocks]
            ).strip()
            return self._render_text_markdown(content)

        rendered_blocks = []
        for block in blocks:
            rendered = self._render_content_block_markdown(block)
            if rendered:
                rendered_blocks.append(rendered)

        if rendered_blocks:
            return (
                '<div class="cc-log"><div class="cc-timeline">\n'
                + "\n".join(rendered_blocks)
                + "\n</div></div>"
            )
        return msg.get("content", "")

    def _render_message_html(self, msg: Dict[str, Any]) -> str:
        """Render a conversation message as safe HTML with structured blocks."""
        blocks = msg.get("content_blocks")
        if not blocks:
            return self._render_text_html(msg.get("content", ""))

        role = msg.get("role", "")
        if role == "assistant":
            return self._render_assistant_turn_html([msg])

        text_only = all(block.get("type") == "text" for block in blocks)
        if text_only:
            content = self._join_text_fragments(
                [block.get("text", "") for block in blocks]
            ).strip()
            return self._render_text_html(content)

        timeline_blocks = []
        text_parts = []
        for block in blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            else:
                rendered = self._render_content_block_markdown(block)
                if rendered:
                    timeline_blocks.append(rendered)

        rendered_parts = []
        if timeline_blocks:
            rendered_parts.append(
                '<div class="cc-log"><div class="cc-timeline">\n'
                + "\n".join(timeline_blocks)
                + "\n</div></div>"
            )

        text = self._join_text_fragments(text_parts).strip()
        if text:
            rendered_parts.append(self._render_text_html(text))

        if rendered_parts:
            return "\n\n".join(rendered_parts)
        return self._render_text_html(msg.get("content", ""))

    def _is_rendered_metadata_message(self, msg: Dict[str, Any]) -> bool:
        """Return True for low-value metadata records that should not be exported."""
        if msg.get("role") == "metadata":
            return True

        blocks = msg.get("content_blocks") or []
        return bool(blocks) and all(
            block.get("type") in self.METADATA_BLOCK_TYPES for block in blocks
        )

    def _render_assistant_turn_markdown(self, messages: List[Dict[str, Any]]) -> str:
        """Render assistant/tool records as one Claude Code-style turn."""
        timeline_blocks = []
        text_parts = []
        pending_tools: List[Dict[str, Any]] = []
        pending_tools_by_id: Dict[str, Dict[str, Any]] = {}

        for msg in messages:
            blocks = msg.get("content_blocks")
            if not blocks:
                content = msg.get("content", "")
                if content and msg.get("role") == "assistant":
                    text_parts.append(content)
                continue

            for block in blocks:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    rendered = self._render_content_block_markdown(block)
                    if rendered:
                        timeline_blocks.append(rendered)
                    pending_tools.append(block)
                    tool_id = block.get("id")
                    if tool_id:
                        pending_tools_by_id[tool_id] = block
                elif block.get("type") == "tool_result":
                    tool_context = None
                    tool_use_id = block.get("tool_use_id")
                    if tool_use_id:
                        tool_context = pending_tools_by_id.pop(tool_use_id, None)
                        pending_tools = [
                            tool
                            for tool in pending_tools
                            if tool.get("id") != tool_use_id
                        ]
                    if tool_context is None and pending_tools:
                        tool_context = pending_tools.pop(0)

                    rendered = self._render_content_block_markdown(block, tool_context)
                    if rendered:
                        timeline_blocks.append(rendered)
                else:
                    rendered = self._render_content_block_markdown(block)
                    if rendered:
                        timeline_blocks.append(rendered)

        parts = []
        if timeline_blocks:
            parts.append(
                '<div class="cc-log"><div class="cc-timeline">\n'
                + "\n".join(timeline_blocks)
                + "\n</div></div>"
            )

        answer = self._join_text_fragments(text_parts).strip()
        if answer:
            answer = self._balance_markdown_fences(answer)
            parts.append(answer)
        return "\n\n".join(parts).strip()

    def _render_assistant_turn_html(self, messages: List[Dict[str, Any]]) -> str:
        """Render assistant/tool records as one HTML turn."""
        timeline_blocks = []
        text_parts = []
        pending_tools: List[Dict[str, Any]] = []
        pending_tools_by_id: Dict[str, Dict[str, Any]] = {}

        for msg in messages:
            blocks = msg.get("content_blocks")
            if not blocks:
                content = msg.get("content", "")
                if content and msg.get("role") == "assistant":
                    text_parts.append(content)
                continue

            for block in blocks:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    rendered = self._render_content_block_markdown(block)
                    if rendered:
                        timeline_blocks.append(rendered)
                    pending_tools.append(block)
                    tool_id = block.get("id")
                    if tool_id:
                        pending_tools_by_id[tool_id] = block
                elif block.get("type") == "tool_result":
                    tool_context = None
                    tool_use_id = block.get("tool_use_id")
                    if tool_use_id:
                        tool_context = pending_tools_by_id.pop(tool_use_id, None)
                        pending_tools = [
                            tool
                            for tool in pending_tools
                            if tool.get("id") != tool_use_id
                        ]
                    if tool_context is None and pending_tools:
                        tool_context = pending_tools.pop(0)

                    rendered = self._render_content_block_markdown(block, tool_context)
                    if rendered:
                        timeline_blocks.append(rendered)
                else:
                    rendered = self._render_content_block_markdown(block)
                    if rendered:
                        timeline_blocks.append(rendered)

        parts = []
        if timeline_blocks:
            parts.append(
                '<div class="cc-log"><div class="cc-timeline">\n'
                + "\n".join(timeline_blocks)
                + "\n</div></div>"
            )

        answer = self._join_text_fragments(text_parts).strip()
        if answer:
            parts.append(self._render_text_html(answer))
        return "\n\n".join(parts).strip()

    def _render_content_block_markdown(
        self, block: Dict[str, Any], tool_context: Optional[Dict[str, Any]] = None
    ) -> str:
        block_type = block.get("type", "unknown")

        if block_type in self.METADATA_BLOCK_TYPES:
            return ""

        if block_type == "text":
            return self._render_text_markdown(block.get("text", ""))

        if block_type == "thinking":
            thinking = block.get("thinking", "").strip()
            if not thinking:
                return ""
            body = self._render_limited_thinking_body(thinking)
            return self._timeline_step(
                "Thinking",
                body,
                css_class="cc-thinking",
                marker="**Thinking**",
            )

        if block_type == "tool_use":
            tool_name = block.get("name", "unknown")
            tool_input = block.get("input", {})
            if tool_name == "TodoWrite":
                return self._render_todo_markdown(tool_input.get("todos", []))

            title = self._tool_title_html(tool_name, tool_input)
            body = self._render_tool_input_markdown(tool_name, tool_input)
            return self._timeline_step(
                title,
                body,
                css_class="cc-tool",
                marker=f"**Tool Use:** `{tool_name}`",
            )

        if block_type == "tool_result":
            result_text = self._tool_result_to_text(block).strip()
            if not result_text:
                return ""
            tool_name = tool_context.get("name", "") if tool_context else ""
            tool_input = tool_context.get("input", {}) if tool_context else {}
            title = "Tool Result: Error" if block.get("is_error") else "Tool Result"
            body = self._render_tool_result_markdown(result_text, tool_name, tool_input)
            return self._timeline_step(
                title,
                body,
                css_class="cc-result",
                marker="**Tool Result:**",
            )

        if block_type == "todo":
            return self._render_todo_markdown(block.get("todos", []))

        if block_type == "image":
            source = block.get("source", {})
            media_type = source.get("media_type", "image")
            return self._timeline_step(
                (
                    'Image <span class="cc-muted">'
                    f"{html.escape(self._single_line_text(media_type))}</span>"
                ),
                "",
                css_class="cc-media",
                marker="**Image:** Image",
            )

        if block_type == "citation":
            title = block.get("title") or block.get("cited_text") or "Citation"
            url = block.get("url")
            if url:
                safe_url = html.escape(self._single_line_text(url), quote=True)
                body = (
                    '<div class="cc-card">'
                    f'<a href="{safe_url}">{self._html_pre_text(title)}</a>'
                    "</div>"
                )
            else:
                body = f'<div class="cc-card">{self._html_pre_text(title)}</div>'
            return self._timeline_step(
                "Citation",
                body,
                css_class="cc-citation",
                marker="**Citation:**",
            )

        if block_type == "api_error":
            status = self._single_line_text(block.get("status", ""))
            title = f"API Error {status}".strip()
            return self._timeline_step(
                title,
                self._render_api_error_markdown(block),
                css_class="cc-api-error",
                marker=f"**API Error:** {status}".strip(),
            )

        label = block_type.replace("_", " ").title()
        payload = json.dumps(block, indent=2, ensure_ascii=False)
        body = f'<div class="cc-unknown">{self._html_code_block(payload)}</div>'
        return self._timeline_step(
            label,
            body,
            css_class="cc-unknown",
            marker=f"**{label}:**",
        )

    def _render_claude_event_markdown(self, block: Dict[str, Any]) -> str:
        summary = self._single_line_text(block.get("summary", ""))
        meta = block.get("metadata") or []
        meta_html = (
            '<div class="cc-event-meta">'
            + "".join(
                f"<span>{html.escape(str(key))}: {html.escape(str(value))}</span>"
                for key, value in meta
            )
            + "</div>"
            if meta
            else ""
        )
        summary_html = (
            f'<div class="cc-event-summary">{html.escape(summary)}</div>'
            if summary
            else ""
        )
        body = (
            '<div class="cc-card cc-event-card">'
            f"{summary_html}{meta_html}</div>"
        )
        details = block.get("details")
        if details:
            payload = json.dumps(details, indent=2, ensure_ascii=False)
            body += (
                '<details class="cc-more-output cc-event-more">'
                "<summary>Show raw event</summary>"
                f"{self._html_code_block(payload, 'json')}"
                "</details>"
            )
        return body

    def _render_file_attachment_markdown(self, block: Dict[str, Any]) -> str:
        metadata = []
        for label, value in (
            ("display", block.get("display_path")),
            ("lines", block.get("num_lines")),
            ("start", block.get("start_line")),
            ("total", block.get("total_lines")),
        ):
            if value not in (None, ""):
                metadata.append(f"{label}: {self._single_line_text(value)}")
        meta_html = (
            '<div class="cc-event-meta">'
            + "".join(f"<span>{html.escape(item)}</span>" for item in metadata)
            + "</div>"
            if metadata
            else ""
        )
        content = block.get("content", "")
        if not content:
            return f'<div class="cc-card cc-event-card">{meta_html}</div>'
        language = self._language_for_path(block.get("filename", ""))
        return meta_html + self._html_code_block(content, language)

    def _render_api_error_markdown(self, block: Dict[str, Any]) -> str:
        summary = block.get("summary") or block.get("message") or "Claude API error"
        meta_items = []
        status = self._single_line_text(block.get("status", ""))
        error_code = self._single_line_text(block.get("error", ""))
        request_id = self._single_line_text(block.get("request_id", ""))
        expected_format = self._single_line_text(block.get("expected_format", ""))
        gateway = self._single_line_text(block.get("gateway", ""))

        if status:
            meta_items.append(f"Status {html.escape(status)}")
        if error_code:
            meta_items.append(html.escape(error_code.replace("_", " ")))
        if request_id:
            meta_items.append(f"Request {html.escape(request_id)}")
        if expected_format:
            meta_items.append(f"Expected {html.escape(expected_format)}")
        if gateway:
            meta_items.append(f"Gateway {html.escape(gateway)}")

        meta_html = (
            '<div class="cc-error-meta">'
            + "".join(f"<span>{item}</span>" for item in meta_items)
            + "</div>"
            if meta_items
            else ""
        )
        fields = block.get("fields") or []
        fields_html = ""
        if fields:
            fields_html = (
                '<div class="cc-error-fields">'
                '<span class="cc-muted">Fields</span>'
                + "".join(
                    f"<code>{html.escape(self._single_line_text(field))}</code>"
                    for field in fields
                )
                + "</div>"
            )

        body = (
            '<div class="cc-card cc-error-card">'
            '<div class="cc-error-summary">'
            f"{html.escape(self._single_line_text(summary))}</div>"
            f"{meta_html}{fields_html}</div>"
        )
        raw_message = str(block.get("message") or "")
        if raw_message:
            body += (
                '<details class="cc-more-output cc-error-more">'
                "<summary>Show raw API error</summary>"
                f"{self._html_code_block(raw_message)}"
                "</details>"
            )

        return body

    def _render_tool_input_markdown(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> str:
        if tool_name in {"Read", "Grep"}:
            return ""

        if tool_name == "Bash":
            command = tool_input.get("command", "")
            return self._io_card("IN", self._html_code_block(command, "bash"))

        if tool_name == "Agent":
            return self._render_agent_input_markdown(tool_input)

        tool_json = json.dumps(tool_input, indent=2, ensure_ascii=False)
        return self._io_card("IN", self._html_code_block(tool_json, "json"))

    def _render_tool_result_markdown(
        self,
        result_text: str,
        tool_name: str = "",
        tool_input: Optional[Dict[str, Any]] = None,
    ) -> str:
        if tool_name == "Read":
            tool_input = tool_input or {}
            file_path = tool_input.get("file_path") or "file content"
            line_label = self._read_lines_label(tool_input)
            line_suffix = (
                f' <span class="cc-muted">({html.escape(line_label)})</span>'
                if line_label
                else ""
            )
            summary = (
                '<span class="cc-read-summary">'
                + self._file_link_html(file_path)
                + line_suffix
                + "</span>"
            )
            return (
                '<details class="cc-card cc-collapse">'
                '<summary><span class="cc-io">OUT</span>'
                f"{summary}</summary>"
                f'{self._html_code_block(result_text, self._language_for_path(file_path))}'
                "</details>"
            )

        if tool_name == "Agent":
            body = (
                '<div class="cc-markdown-block">'
                + self._simple_markdown_html(result_text)
                + "</div>"
            )
            return self._io_card("OUT", body, css_class="cc-agent-card")

        return self._render_limited_tool_output(result_text)

    def _render_todo_markdown(self, todos: Any) -> str:
        if not isinstance(todos, list) or not todos:
            return ""

        items = []
        for todo in todos:
            if not isinstance(todo, dict):
                continue
            status = todo.get("status", "pending")
            content = todo.get("content") or todo.get("activeForm") or ""
            if not content:
                continue
            css_status = status.replace("_", "-")
            check = "✓" if status == "completed" else ("*" if status == "in_progress" else "")
            status_label = "in progress" if status == "in_progress" else status
            content_label = self._single_line_text(content)
            items.append(
                f'<li class="cc-todo-item cc-{css_status}">'
                f'<span class="cc-checkbox">{html.escape(check)}</span>'
                f"<span>{html.escape(content_label)}</span>"
                f'<span class="cc-muted">({html.escape(status_label)})</span>'
                "</li>"
            )

        if not items:
            return ""

        body = '<ul class="cc-todo-list">' + "\n".join(items) + "</ul>"
        return self._timeline_step(
            "Update Todos",
            body,
            css_class="cc-tool cc-todo",
            marker="**Todo List**",
        )

    def _timeline_step(
        self,
        title: str,
        body: str,
        css_class: str = "",
        details: bool = False,
        marker: str = "",
    ) -> str:
        marker_attr = (
            f' data-markdown-label="{html.escape(marker, quote=True)}"'
            if marker
            else ""
        )
        title_html = title if "<" in title else html.escape(title)

        if details:
            return (
                f'<details class="cc-step {css_class}"{marker_attr}>'
                f'<summary><span class="cc-dot"></span>'
                f'<span class="cc-title">{title_html}</span></summary>'
                f'<div class="cc-body">{body}</div></details>'
            )

        body_html = f'<div class="cc-body">{body}</div>' if body else ""
        return (
            f'<div class="cc-step {css_class}"{marker_attr}>'
            '<span class="cc-dot"></span>'
            f'<div class="cc-title">{title_html}</div>{body_html}</div>'
        )

    def _tool_title_html(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        secondary = ""
        if tool_name == "Bash":
            secondary = tool_input.get("description") or tool_input.get("command", "")
        elif tool_name == "Read":
            file_path = tool_input.get("file_path", "")
            line_label = self._read_lines_label(tool_input)
            line_suffix = (
                f' <span class="cc-muted">({html.escape(line_label)})</span>'
                if line_label
                else ""
            )
            return f"{html.escape(tool_name)} {self._file_link_html(file_path)}{line_suffix}"
        elif tool_name in {"Write", "Edit", "MultiEdit"}:
            secondary = tool_input.get("file_path", "")
        elif tool_name == "Grep":
            pattern = self._single_line_text(tool_input.get("pattern", ""))
            path = self._single_line_text(tool_input.get("path", ""))
            pattern_html = (
                f'"{html.escape(pattern)}"' if pattern else html.escape(tool_name)
            )
            if path:
                return (
                    f"{html.escape(tool_name)} {pattern_html} "
                    f'<span class="cc-muted">(in {html.escape(path)})</span>'
                )
            return f"{html.escape(tool_name)} {pattern_html}"
        elif tool_name == "Glob":
            secondary = tool_input.get("pattern", "")
        elif tool_name in {"WebFetch", "WebSearch"}:
            secondary = tool_input.get("url") or tool_input.get("query", "")
        elif tool_name == "Agent":
            secondary = tool_input.get("description") or tool_input.get("prompt", "")
        else:
            secondary = (
                tool_input.get("description")
                or tool_input.get("prompt")
                or tool_input.get("query")
                or ""
            )

        if secondary:
            secondary_label = self._single_line_text(secondary)[:120]
            return (
                f"{html.escape(tool_name)} "
                f'<span class="cc-muted">{html.escape(secondary_label)}</span>'
            )
        return html.escape(tool_name)

    def _claude_icon_data_uri(self) -> str:
        icon_paths = [self.CLAUDE_CODE_ICON]
        for extension_dir in (
            Path.home() / ".vscode-server" / "extensions",
            Path.home() / ".vscode" / "extensions",
        ):
            if extension_dir.exists():
                icon_paths.extend(
                    sorted(
                        extension_dir.glob(
                            "anthropic.claude-code-*/resources/claude-logo.png"
                        ),
                        reverse=True,
                    )
                )

        for icon_path in icon_paths:
            try:
                icon_bytes = icon_path.read_bytes()
                encoded = base64.b64encode(icon_bytes).decode("ascii")
                return f"data:image/png;base64,{encoded}"
            except Exception:
                continue
        return ""

    def display_conversation(self, jsonl_path: Path, detailed: bool = False) -> None:
        """Display a conversation in the terminal with pagination.
        
        Args:
            jsonl_path: Path to the JSONL file
            detailed: If True, include tool use and system messages
        """
        try:
            # Extract conversation
            messages = self.extract_conversation(jsonl_path, detailed=detailed)
            
            if not messages:
                print("❌ No messages found in conversation")
                return
            
            # Get session info
            session_id = jsonl_path.stem
            
            # Clear screen and show header
            print("\033[2J\033[H", end="")  # Clear screen
            print("=" * 60)
            print(f"📄 Viewing: {jsonl_path.parent.name}")
            print(f"Session: {session_id[:8]}...")
            
            # Get timestamp from first message
            first_timestamp = messages[0].get("timestamp", "")
            if first_timestamp:
                try:
                    dt = datetime.fromisoformat(first_timestamp.replace("Z", "+00:00"))
                    print(f"Date: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception:
                    pass
            
            print("=" * 60)
            print("↑↓ to scroll • Q to quit • Enter to continue\n")
            
            # Display messages with pagination
            lines_shown = 8  # Header lines
            lines_per_page = 30
            
            for i, msg in enumerate(messages):
                role = msg["role"]
                content = msg["content"]
                
                # Format role display
                if role == "user" or role == "human":
                    print(f"\n{'─' * 40}")
                    print(f"👤 HUMAN:")
                    print(f"{'─' * 40}")
                elif role == "assistant":
                    print(f"\n{'─' * 40}")
                    print(f"🤖 CLAUDE:")
                    print(f"{'─' * 40}")
                elif role == "tool_use":
                    print(f"\n🔧 TOOL USE:")
                elif role == "tool_result":
                    print(f"\n📤 TOOL RESULT:")
                elif role == "system":
                    print(f"\nℹ️ SYSTEM:")
                else:
                    print(f"\n{role.upper()}:")
                
                # Display content (limit very long messages)
                lines = content.split('\n')
                max_lines_per_msg = 50
                
                for line_idx, line in enumerate(lines[:max_lines_per_msg]):
                    # Wrap very long lines
                    if len(line) > 100:
                        line = line[:97] + "..."
                    print(line)
                    lines_shown += 1
                    
                    # Check if we need to paginate
                    if lines_shown >= lines_per_page:
                        response = input("\n[Enter] Continue • [Q] Quit: ").strip().upper()
                        if response == "Q":
                            print("\n👋 Stopped viewing")
                            return
                        # Clear screen for next page
                        print("\033[2J\033[H", end="")
                        lines_shown = 0
                
                if len(lines) > max_lines_per_msg:
                    print(f"... [{len(lines) - max_lines_per_msg} more lines truncated]")
                    lines_shown += 1
            
            print("\n" + "=" * 60)
            print("📄 End of conversation")
            print("=" * 60)
            input("\nPress Enter to continue...")
            
        except Exception as e:
            print(f"❌ Error displaying conversation: {e}")
            input("\nPress Enter to continue...")

    def save_as_markdown(
        self, conversation: List[Dict[str, Any]], session_id: str
    ) -> Optional[Path]:
        """Save conversation as clean markdown file."""
        if not conversation:
            return None

        # Get timestamp from first message
        first_timestamp = conversation[0].get("timestamp", "")
        if first_timestamp:
            try:
                # Parse ISO timestamp
                dt = datetime.fromisoformat(first_timestamp.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
            except Exception:
                date_str = datetime.now().strftime("%Y-%m-%d")
                time_str = ""
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = ""

        filename = f"claude-conversation-{date_str}-{session_id[:8]}.md"
        output_path = self.output_dir / filename
        icon_uri = self._claude_icon_data_uri()
        claude_heading = (
            '<h2 class="cc-claude-heading" data-markdown-label="## 🤖 Claude">Claude</h2>'
            if icon_uri
            else "## 🤖 Claude"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Claude Conversation Log\n\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Date: {date_str}")
            if time_str:
                f.write(f" {time_str}")
            f.write("\n\n")
            f.write(self._markdown_style_block(icon_uri))
            f.write("\n---\n\n")

            render_messages = [
                msg for msg in conversation if not self._is_rendered_metadata_message(msg)
            ]
            assistant_roles = {"assistant", "tool_use", "tool_result", "todo"}
            i = 0
            while i < len(render_messages):
                msg = render_messages[i]
                role = msg["role"]
                
                if role == "user":
                    content = self._render_message_markdown(msg)
                    f.write("## 👤 User\n\n")
                    f.write(f"{content}\n\n")
                    f.write("---\n\n")
                    i += 1
                elif role in assistant_roles:
                    turn_messages = []
                    while (
                        i < len(render_messages)
                        and render_messages[i]["role"] in assistant_roles
                    ):
                        turn_messages.append(render_messages[i])
                        i += 1
                    content = self._render_assistant_turn_markdown(turn_messages)
                    if content:
                        f.write(f"{claude_heading}\n\n")
                        f.write(f"{content}\n\n")
                        f.write("---\n\n")
                elif role == "system":
                    content = self._render_message_markdown(msg)
                    f.write("### ℹ️ System\n\n")
                    f.write(f"{content}\n\n")
                    f.write("---\n\n")
                    i += 1
                else:
                    content = self._render_message_markdown(msg)
                    f.write(f"## {role}\n\n")
                    f.write(f"{content}\n\n")
                    f.write("---\n\n")
                    i += 1

        return output_path
    
    def save_as_json(
        self, conversation: List[Dict[str, Any]], session_id: str
    ) -> Optional[Path]:
        """Save conversation as JSON file."""
        if not conversation:
            return None

        # Get timestamp from first message
        first_timestamp = conversation[0].get("timestamp", "")
        if first_timestamp:
            try:
                dt = datetime.fromisoformat(first_timestamp.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                date_str = datetime.now().strftime("%Y-%m-%d")
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")

        filename = f"claude-conversation-{date_str}-{session_id[:8]}.json"
        output_path = self.output_dir / filename

        # Create JSON structure
        output = {
            "session_id": session_id,
            "date": date_str,
            "message_count": len(conversation),
            "messages": conversation
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return output_path
    
    def save_as_html(
        self, conversation: List[Dict[str, Any]], session_id: str
    ) -> Optional[Path]:
        """Save conversation as HTML file with syntax highlighting."""
        if not conversation:
            return None

        # Get timestamp from first message
        first_timestamp = conversation[0].get("timestamp", "")
        if first_timestamp:
            try:
                dt = datetime.fromisoformat(first_timestamp.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
            except Exception:
                date_str = datetime.now().strftime("%Y-%m-%d")
                time_str = ""
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = ""

        filename = f"claude-conversation-{date_str}-{session_id[:8]}.html"
        output_path = self.output_dir / filename
        icon_uri = self._claude_icon_data_uri()
        icon_html = (
            f'<img src="{icon_uri}" alt="Claude Code" class="claude-icon">'
            if icon_uri
            else '<span class="claude-dot"></span>'
        )
        small_icon_html = (
            f'<img src="{icon_uri}" alt="" class="role-icon">'
            if icon_uri
            else '<span class="role-dot"></span>'
        )
        render_messages = [
            msg for msg in conversation if not self._is_rendered_metadata_message(msg)
        ]

        # HTML template with dark Claude Code-style rendering.
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claude Conversation - {session_id[:8]}</title>
    <style>
        :root {{
            --html-bg: #0f1117;
            --html-panel: #171a21;
            --html-panel-2: #1f232b;
            --html-border: #343946;
            --html-fg: #e7e9ee;
            --html-muted: #9ca3af;
            --html-blue: #4daafc;
            --html-green: #4ade80;
            --html-orange: #f59e0b;
            --html-red: #f87171;
            --html-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--html-fg);
            max-width: 1080px;
            margin: 0 auto;
            padding: 28px 20px 40px;
            background: var(--html-bg);
        }}
        .header {{
            background: linear-gradient(180deg, #1b1f27 0%, var(--html-panel) 100%);
            border: 1px solid var(--html-border);
            padding: 22px 24px;
            border-radius: 8px;
            margin-bottom: 18px;
            box-shadow: 0 18px 36px rgba(0,0,0,0.26);
        }}
        h1 {{
            color: var(--html-fg);
            font-size: 1.55rem;
            line-height: 1.2;
            margin: 0 0 14px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .metadata {{
            color: var(--html-muted);
            display: flex;
            flex-wrap: wrap;
            font-size: 0.84em;
            gap: 8px;
        }}
        .metadata p {{
            border: 1px solid var(--html-border);
            border-radius: 999px;
            margin: 0;
            padding: 3px 9px;
        }}
        .claude-icon {{
            width: 32px;
            height: 32px;
            border-radius: 50%;
        }}
        .role-icon {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            flex: 0 0 auto;
        }}
        .claude-dot,
        .role-dot {{
            display: inline-block;
            background: #d97757;
            border-radius: 50%;
        }}
        .claude-dot {{
            width: 32px;
            height: 32px;
        }}
        .role-dot {{
            width: 18px;
            height: 18px;
        }}
        .message {{
            background: var(--html-panel);
            border: 1px solid var(--html-border);
            padding: 16px 18px 18px;
            margin-bottom: 14px;
            border-radius: 8px;
            box-shadow: 0 10px 26px rgba(0,0,0,0.18);
        }}
        .user {{
            border-left: 4px solid var(--html-blue);
        }}
        .assistant {{
            border-left: 4px solid var(--html-green);
        }}
        .tool_use {{
            border-left: 4px solid var(--html-orange);
        }}
        .tool_result {{
            border-left: 4px solid var(--html-red);
        }}
        .system {{
            border-left: 4px solid var(--html-muted);
        }}
        .role {{
            font-weight: bold;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            letter-spacing: 0;
        }}
        .content {{
            white-space: normal;
            word-wrap: break-word;
        }}
        .content > *:first-child {{
            margin-top: 0;
        }}
        .content > *:last-child {{
            margin-bottom: 0;
        }}
        .content h2,
        .content h3,
        .content h4,
        .content h5,
        .content h6 {{
            color: var(--html-fg);
            font-weight: 650;
            line-height: 1.25;
            margin: 1.1rem 0 0.55rem;
        }}
        .content h2:first-child,
        .content h3:first-child,
        .content h4:first-child,
        .content h5:first-child,
        .content h6:first-child {{
            margin-top: 0;
        }}
        .content h2 {{
            font-size: 1.65rem;
        }}
        .content h3 {{
            font-size: 1.36rem;
        }}
        .content h4 {{
            font-size: 1.14rem;
        }}
        .content h5 {{
            font-size: 1.02rem;
        }}
        .content h6 {{
            font-size: 0.94rem;
        }}
        pre {{
            background: var(--html-panel-2);
            border: 1px solid var(--html-border);
            color: var(--html-fg);
            font-family: var(--html-mono);
            font-size: 0.92em;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
        }}
        pre code {{
            background: transparent;
            display: block;
            padding: 0;
        }}
        code {{
            background: var(--html-panel-2);
            color: var(--html-fg);
            padding: 2px 5px;
            border-radius: 3px;
            font-family: var(--html-mono);
        }}
        a {{
            color: var(--html-blue);
        }}
        .MathJax,
        mjx-container {{
            color: var(--html-fg);
        }}
        mjx-container[jax="CHTML"] {{
            max-width: 100%;
            overflow: visible;
            padding: 0.08em 0;
        }}
        .cc-math-block mjx-container[jax="CHTML"] {{
            max-width: none;
            min-width: max-content;
        }}
        mjx-container[jax="CHTML"][display="true"] {{
            margin: 0.25rem 0;
            padding: 0.18rem 0;
        }}
        @media (max-width: 720px) {{
            body {{
                padding: 14px 10px 24px;
            }}
            .header,
            .message {{
                padding: 14px;
            }}
            h1 {{
                font-size: 1.25rem;
            }}
            .metadata {{
                gap: 6px;
            }}
        }}
    </style>
    {self._markdown_style_block(icon_uri)}
    <script>
        window.MathJax = {{
            loader: {{
                load: ['[tex]/ams', '[tex]/textmacros']
            }},
            tex: {{
                packages: {{ '[+]': ['ams', 'textmacros'] }},
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true,
                processEnvironments: true
            }},
            options: {{
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
            }}
        }};
    </script>
    <script>
        document.addEventListener('DOMContentLoaded', () => {{
            document.addEventListener('click', async (event) => {{
                const button = event.target.closest('.cc-copy-code');
                if (!button) {{
                    return;
                }}

                const code = button.closest('.cc-code-block')?.querySelector('pre code');
                if (!code) {{
                    return;
                }}

                const original = button.textContent;
                const text = code.textContent.replace(/\\n$/, '');

                try {{
                    if (navigator.clipboard && window.isSecureContext) {{
                        await navigator.clipboard.writeText(text);
                    }} else {{
                        const textarea = document.createElement('textarea');
                        textarea.value = text;
                        textarea.setAttribute('readonly', '');
                        textarea.style.position = 'fixed';
                        textarea.style.left = '-9999px';
                        document.body.appendChild(textarea);
                        textarea.select();
                        document.execCommand('copy');
                        document.body.removeChild(textarea);
                    }}

                    button.textContent = 'Copied';
                }} catch (error) {{
                    button.textContent = 'Copy failed';
                }}

                window.setTimeout(() => {{
                    button.textContent = original || 'Copy';
                }}, 1200);
            }});
        }});
    </script>
    <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
    <div class="header">
        <h1>{icon_html} Claude Conversation Log</h1>
        <div class="metadata">
            <p>Session ID: {html.escape(session_id)}</p>
            <p>Date: {date_str} {time_str}</p>
            <p>Messages: {len(render_messages)}</p>
        </div>
    </div>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

            assistant_roles = {"assistant", "tool_use", "tool_result", "todo"}
            i = 0
            while i < len(render_messages):
                msg = render_messages[i]
                role = msg["role"]

                if role == "user":
                    content = self._render_message_html(msg)
                    role_display = "👤 User"
                    css_role = "user"
                    i += 1
                elif role in assistant_roles:
                    turn_messages = []
                    while (
                        i < len(render_messages)
                        and render_messages[i]["role"] in assistant_roles
                    ):
                        turn_messages.append(render_messages[i])
                        i += 1
                    content = self._render_assistant_turn_html(turn_messages)
                    role_display = f"{small_icon_html} Claude"
                    css_role = "assistant"
                    if not content:
                        continue
                elif role == "system":
                    content = self._render_message_html(msg)
                    role_display = "ℹ️ System"
                    css_role = "system"
                    i += 1
                else:
                    content = self._render_message_html(msg)
                    role_display = html.escape(role)
                    css_role = html.escape(role)
                    i += 1

                f.write(f'    <div class="message {css_role}">\n')
                f.write(f'        <div class="role">{role_display}</div>\n')
                f.write(f'        <div class="content">{content}</div>\n')
                f.write("    </div>\n")

            f.write("\n</body>\n</html>")

        return output_path

    def save_conversation(
        self, conversation: List[Dict[str, Any]], session_id: str, format: str = "markdown"
    ) -> Optional[Path]:
        """Save conversation in the specified format.
        
        Args:
            conversation: The conversation data
            session_id: Session identifier
            format: Output format ('markdown', 'json', 'html')
        """
        if format == "markdown":
            return self.save_as_markdown(conversation, session_id)
        elif format == "json":
            return self.save_as_json(conversation, session_id)
        elif format == "html":
            return self.save_as_html(conversation, session_id)
        else:
            print(f"❌ Unsupported format: {format}")
            return None

    def get_conversation_preview(self, session_path: Path) -> Tuple[str, int]:
        """Get a preview of the conversation's first real user message and message count."""
        try:
            first_user_msg = ""
            msg_count = 0
            
            with open(session_path, 'r', encoding='utf-8') as f:
                for line in f:
                    msg_count += 1
                    if not first_user_msg:
                        try:
                            data = json.loads(line)
                            # Check for user message
                            if data.get("type") == "user" and "message" in data:
                                msg = data["message"]
                                if msg.get("role") == "user":
                                    content = msg.get("content", "")
                                    
                                    # Handle list content (common format in Claude JSONL)
                                    if isinstance(content, list):
                                        for item in content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                text = item.get("text", "").strip()
                                                
                                                # Skip tool results
                                                if text.startswith("tool_use_id"):
                                                    continue
                                                
                                                # Skip interruption messages
                                                if "[Request interrupted" in text:
                                                    continue
                                                
                                                # Skip Claude's session continuation messages
                                                if "session is being continued" in text.lower():
                                                    continue
                                                
                                                # Remove XML-like tags (command messages, etc)
                                                import re
                                                text = re.sub(r'<[^>]+>', '', text).strip()
                                                
                                                # Skip command outputs  
                                                if "is running" in text and "…" in text:
                                                    continue
                                                
                                                # Handle image references - extract text after them
                                                if text.startswith("[Image #"):
                                                    parts = text.split("]", 1)
                                                    if len(parts) > 1:
                                                        text = parts[1].strip()
                                                
                                                # If we have real user text, use it
                                                if text and len(text) > 3:  # Lower threshold to catch "hello"
                                                    first_user_msg = text[:100].replace('\n', ' ')
                                                    break
                                    
                                    # Handle string content (less common but possible)
                                    elif isinstance(content, str):
                                        import re
                                        content = content.strip()
                                        
                                        # Remove XML-like tags
                                        content = re.sub(r'<[^>]+>', '', content).strip()
                                        
                                        # Skip command outputs
                                        if "is running" in content and "…" in content:
                                            continue
                                        
                                        # Skip Claude's session continuation messages
                                        if "session is being continued" in content.lower():
                                            continue
                                        
                                        # Skip tool results and interruptions
                                        if not content.startswith("tool_use_id") and "[Request interrupted" not in content:
                                            if content and len(content) > 3:  # Lower threshold to catch short messages
                                                first_user_msg = content[:100].replace('\n', ' ')
                        except json.JSONDecodeError:
                            continue
                            
            return first_user_msg or "No preview available", msg_count
        except Exception as e:
            return f"Error: {str(e)[:30]}", 0

    def list_recent_sessions(self, limit: int = None) -> List[Path]:
        """List recent sessions with details."""
        sessions = self.find_sessions()

        if not sessions:
            print("❌ No Claude sessions found in ~/.claude/projects/")
            print("💡 Make sure you've used Claude Code and have conversations saved.")
            return []

        print(f"\n📚 Found {len(sessions)} Claude sessions:\n")
        print("=" * 80)

        # Show all sessions if no limit specified
        sessions_to_show = sessions[:limit] if limit else sessions
        for i, session in enumerate(sessions_to_show, 1):
            # Clean up project name (remove hyphens, make readable)
            project = session.parent.name.replace('-', ' ').strip()
            if project.startswith("Users"):
                project = "~/" + "/".join(project.split()[2:]) if len(project.split()) > 2 else "Home"
            
            session_id = session.stem
            modified = datetime.fromtimestamp(session.stat().st_mtime)

            # Get file size
            size = session.stat().st_size
            size_kb = size / 1024
            
            # Get preview and message count
            preview, msg_count = self.get_conversation_preview(session)

            # Print formatted info
            print(f"\n{i}. 📁 {project}")
            print(f"   📄 Session: {session_id[:8]}...")
            print(f"   📅 Modified: {modified.strftime('%Y-%m-%d %H:%M')}")
            print(f"   💬 Messages: {msg_count}")
            print(f"   💾 Size: {size_kb:.1f} KB")
            print(f"   📝 Preview: \"{preview}...\"")

        print("\n" + "=" * 80)
        return sessions[:limit]

    def extract_multiple(
        self, sessions: List[Path], indices: List[int], 
        format: str = "markdown", detailed: bool = False
    ) -> Tuple[int, int]:
        """Extract multiple sessions by index.
        
        Args:
            sessions: List of session paths
            indices: Indices to extract
            format: Output format ('markdown', 'json', 'html')
            detailed: If True, include tool use and system messages
        """
        success = 0
        total = len(indices)

        for idx in indices:
            if 0 <= idx < len(sessions):
                session_path = sessions[idx]
                conversation = self.extract_conversation(session_path, detailed=detailed)
                if conversation:
                    output_path = self.save_conversation(conversation, session_path.stem, format=format)
                    success += 1
                    msg_count = len(conversation)
                    print(
                        f"✅ {success}/{total}: {output_path.name} "
                        f"({msg_count} messages)"
                    )
                else:
                    print(f"⏭️  Skipped session {idx + 1} (no conversation)")
            else:
                print(f"❌ Invalid session number: {idx + 1}")

        return success, total


def main():
    parser = argparse.ArgumentParser(
        description="Extract Claude Code conversations to clean markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                    # List all available sessions
  %(prog)s --extract 1               # Extract the most recent session
  %(prog)s --extract 1,3,5           # Extract specific sessions
  %(prog)s --recent 5                # Extract 5 most recent sessions
  %(prog)s --all                     # Extract all sessions
  %(prog)s --output ~/my-logs        # Specify output directory
  %(prog)s --search "python error"   # Search conversations
  %(prog)s --search-regex "import.*" # Search with regex
  %(prog)s --format json --all       # Export all as JSON
  %(prog)s --format html --extract 1 # Export session 1 as HTML
  %(prog)s --detailed --extract 1    # Include tool use & system messages
        """,
    )
    parser.add_argument("--list", action="store_true", help="List recent sessions")
    parser.add_argument(
        "--extract",
        type=str,
        help="Extract specific session(s) by number (comma-separated)",
    )
    parser.add_argument(
        "--all", "--logs", action="store_true", help="Extract all sessions"
    )
    parser.add_argument(
        "--recent", type=int, help="Extract N most recent sessions", default=0
    )
    parser.add_argument(
        "--output", type=str, help="Output directory for markdown files"
    )
    parser.add_argument(
        "--limit", type=int, help="Limit for --list command (default: show all)", default=None
    )
    parser.add_argument(
        "--interactive",
        "-i",
        "--start",
        "-s",
        action="store_true",
        help="Launch interactive UI for easy extraction",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export mode: 'logs' for interactive UI",
    )

    # Search arguments
    parser.add_argument(
        "--search", type=str, help="Search conversations for text (smart search)"
    )
    parser.add_argument(
        "--search-regex", type=str, help="Search conversations using regex pattern"
    )
    parser.add_argument(
        "--search-date-from", type=str, help="Filter search from date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--search-date-to", type=str, help="Filter search to date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--search-speaker",
        choices=["human", "assistant", "both"],
        default="both",
        help="Filter search by speaker",
    )
    parser.add_argument(
        "--case-sensitive", action="store_true", help="Make search case-sensitive"
    )
    
    # Export format arguments
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "html"],
        default="markdown",
        help="Output format for exported conversations (default: markdown)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include tool use, MCP responses, and system messages in export"
    )

    args = parser.parse_args()

    # Handle interactive mode
    if args.interactive or (args.export and args.export.lower() == "logs"):
        from interactive_ui import main as interactive_main

        interactive_main()
        return

    # Initialize extractor with optional output directory
    extractor = ClaudeConversationExtractor(args.output)

    # Handle search mode
    if args.search or args.search_regex:
        from datetime import datetime

        from search_conversations import ConversationSearcher

        searcher = ConversationSearcher()

        # Determine search mode and query
        if args.search_regex:
            query = args.search_regex
            mode = "regex"
        else:
            query = args.search
            mode = "smart"

        # Parse date filters
        date_from = None
        date_to = None
        if args.search_date_from:
            try:
                date_from = datetime.strptime(args.search_date_from, "%Y-%m-%d")
            except ValueError:
                print(f"❌ Invalid date format: {args.search_date_from}")
                return

        if args.search_date_to:
            try:
                date_to = datetime.strptime(args.search_date_to, "%Y-%m-%d")
            except ValueError:
                print(f"❌ Invalid date format: {args.search_date_to}")
                return

        # Speaker filter
        speaker_filter = None if args.search_speaker == "both" else args.search_speaker

        # Perform search
        print(f"🔍 Searching for: {query}")
        results = searcher.search(
            query=query,
            mode=mode,
            date_from=date_from,
            date_to=date_to,
            speaker_filter=speaker_filter,
            case_sensitive=args.case_sensitive,
            max_results=30,
        )

        if not results:
            print("❌ No matches found.")
            return

        print(f"\n✅ Found {len(results)} matches across conversations:")

        # Group and display results
        results_by_file = {}
        for result in results:
            if result.file_path not in results_by_file:
                results_by_file[result.file_path] = []
            results_by_file[result.file_path].append(result)

        # Store file paths for potential viewing
        file_paths_list = []
        for file_path, file_results in results_by_file.items():
            file_paths_list.append(file_path)
            print(f"\n{len(file_paths_list)}. 📄 {file_path.parent.name} ({len(file_results)} matches)")
            # Show first match preview
            first = file_results[0]
            print(f"   {first.speaker}: {first.matched_content[:100]}...")

        # Offer to view conversations
        if file_paths_list:
            print("\n" + "=" * 60)
            try:
                view_choice = input("\nView a conversation? Enter number (1-{}) or press Enter to skip: ".format(
                    len(file_paths_list))).strip()
                
                if view_choice.isdigit():
                    view_num = int(view_choice)
                    if 1 <= view_num <= len(file_paths_list):
                        selected_path = file_paths_list[view_num - 1]
                        extractor.display_conversation(selected_path, detailed=args.detailed)
                        
                        # Offer to extract after viewing
                        extract_choice = input("\n📤 Extract this conversation? (y/N): ").strip().lower()
                        if extract_choice == 'y':
                            conversation = extractor.extract_conversation(selected_path, detailed=args.detailed)
                            if conversation:
                                session_id = selected_path.stem
                                if args.format == "json":
                                    output = extractor.save_as_json(conversation, session_id)
                                elif args.format == "html":
                                    output = extractor.save_as_html(conversation, session_id)
                                else:
                                    output = extractor.save_as_markdown(conversation, session_id)
                                print(f"✅ Saved: {output.name}")
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Cancelled")
        
        return

    # Default action is to list sessions
    if args.list or (
        not args.extract
        and not args.all
        and not args.recent
        and not args.search
        and not args.search_regex
    ):
        sessions = extractor.list_recent_sessions(args.limit)

        if sessions and not args.list:
            print("\nTo extract conversations:")
            print("  claude-extract --extract <number>      # Extract specific session")
            print("  claude-extract --recent 5              # Extract 5 most recent")
            print("  claude-extract --all                   # Extract all sessions")

    elif args.extract:
        sessions = extractor.find_sessions()

        # Parse comma-separated indices
        indices = []
        for num in args.extract.split(","):
            try:
                idx = int(num.strip()) - 1  # Convert to 0-based index
                indices.append(idx)
            except ValueError:
                print(f"❌ Invalid session number: {num}")
                continue

        if indices:
            print(f"\n📤 Extracting {len(indices)} session(s) as {args.format.upper()}...")
            if args.detailed:
                print("📋 Including detailed tool use and system messages")
            success, total = extractor.extract_multiple(
                sessions, indices, format=args.format, detailed=args.detailed
            )
            print(f"\n✅ Successfully extracted {success}/{total} sessions")

    elif args.recent:
        sessions = extractor.find_sessions()
        limit = min(args.recent, len(sessions))
        print(f"\n📤 Extracting {limit} most recent sessions as {args.format.upper()}...")
        if args.detailed:
            print("📋 Including detailed tool use and system messages")

        indices = list(range(limit))
        success, total = extractor.extract_multiple(
            sessions, indices, format=args.format, detailed=args.detailed
        )
        print(f"\n✅ Successfully extracted {success}/{total} sessions")

    elif args.all:
        sessions = extractor.find_sessions()
        print(f"\n📤 Extracting all {len(sessions)} sessions as {args.format.upper()}...")
        if args.detailed:
            print("📋 Including detailed tool use and system messages")

        indices = list(range(len(sessions)))
        success, total = extractor.extract_multiple(
            sessions, indices, format=args.format, detailed=args.detailed
        )
        print(f"\n✅ Successfully extracted {success}/{total} sessions")


def launch_interactive():
    """Launch the interactive UI directly, or handle search if specified."""
    import sys
    
    # If no arguments provided, launch interactive UI
    if len(sys.argv) == 1:
        try:
            from .interactive_ui import main as interactive_main
        except ImportError:
            from interactive_ui import main as interactive_main
        interactive_main()
    # Check if 'search' was passed as an argument
    elif len(sys.argv) > 1 and sys.argv[1] == 'search':
        # Launch real-time search with viewing capability
        try:
            from .realtime_search import RealTimeSearch, create_smart_searcher
            from .search_conversations import ConversationSearcher
        except ImportError:
            from realtime_search import RealTimeSearch, create_smart_searcher
            from search_conversations import ConversationSearcher
        
        # Initialize components
        extractor = ClaudeConversationExtractor()
        searcher = ConversationSearcher()
        smart_searcher = create_smart_searcher(searcher)
        
        # Run search
        rts = RealTimeSearch(smart_searcher, extractor)
        selected_file = rts.run()
        
        if selected_file:
            # View the selected conversation
            extractor.display_conversation(selected_file)
            
            # Offer to extract
            try:
                extract_choice = input("\n📤 Extract this conversation? (y/N): ").strip().lower()
                if extract_choice == 'y':
                    conversation = extractor.extract_conversation(selected_file)
                    if conversation:
                        session_id = selected_file.stem
                        output = extractor.save_as_markdown(conversation, session_id)
                        print(f"✅ Saved: {output.name}")
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Cancelled")
    else:
        # If other arguments are provided, run the normal CLI
        main()


if __name__ == "__main__":
    main()
