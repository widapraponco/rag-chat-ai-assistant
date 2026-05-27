"""
Vanna 2.0 Response Stream Handler
==================================
Parse Vanna 2.0 streaming response into readable chat UI components.
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    STATUS_BAR = "status_bar_update"
    TASK_TRACKER = "task_tracker_update"
    CHAT_INPUT = "chat_input_update"
    TEXT = "text"
    UNKNOWN = "unknown"

class TaskOperation(Enum):
    ADD = "add_task"
    UPDATE = "update_task"
    REMOVE = "remove_task"

@dataclass
class ParsedEvent:
    raw_type: str
    event_type: EventType
    timestamp: str
    id: str
    lifecycle: str
    data: Dict[str, Any]
    simple_component: Optional[Dict] = None

    @property
    def formatted_time(self) -> str:
        try:
            dt = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            return dt.strftime('%H:%M:%S')
        except:
            return self.timestamp

@dataclass  
class ChatMessage:
    content: str
    is_markdown: bool
    component_id: str
    timestamp: str
    simple_text: Optional[str] = None

@dataclass
class TaskInfo:
    task_id: str
    title: str
    description: str
    status: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

class VannaStreamHandler:
    """
    Handles Vanna 2.0 streaming response for chat UI rendering.

    Usage:
        handler = VannaStreamHandler()
        for line in stream:
            handler.parse_line(line)

        # Get final readable content
        messages = handler.get_chat_messages()
        status = handler.get_current_status()
        tasks = handler.get_tasks()
    """

    def __init__(self):
        self.events: List[ParsedEvent] = []
        self.tasks: Dict[str, TaskInfo] = {}
        self.current_status: Optional[Dict] = None
        self.chat_messages: List[ChatMessage] = []
        self.is_processing: bool = False

    def parse_line(self, line: str) -> Optional[ParsedEvent]:
        """Parse a single line from Vanna stream."""
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError:
            return None

        rc = data.get('rich_component', {})
        if not rc:
            return None

        event_type = self._detect_type(rc.get('type', ''))

        event = ParsedEvent(
            raw_type=rc.get('type', ''),
            event_type=event_type,
            timestamp=data.get('timestamp', ''),
            id=rc.get('id', ''),
            lifecycle=rc.get('lifecycle', ''),
            data=rc,
            simple_component=data.get('simple_component')
        )

        self.events.append(event)
        self._process_event(event)
        return event

    def _detect_type(self, type_str: str) -> EventType:
        try:
            return EventType(type_str)
        except ValueError:
            return EventType.UNKNOWN

    def _process_event(self, event: ParsedEvent):
        """Process event and update internal state."""

        if event.event_type == EventType.STATUS_BAR:
            self.current_status = {
                'status': event.data.get('status'),
                'message': event.data.get('message'),
                'detail': event.data.get('detail'),
                'timestamp': event.timestamp
            }
            self.is_processing = event.data.get('status') == 'working'

        elif event.event_type == EventType.TASK_TRACKER:
            self._handle_task_event(event)

        elif event.event_type == EventType.TEXT:
            msg = ChatMessage(
                content=event.data.get('content', ''),
                is_markdown=event.data.get('markdown', False),
                component_id=event.data.get('id', ''),
                timestamp=event.timestamp,
                simple_text=event.simple_component.get('text') if event.simple_component else None
            )
            self.chat_messages.append(msg)

        elif event.event_type == EventType.CHAT_INPUT:
            # Input state update - usually means response is complete
            pass

    def _handle_task_event(self, event: ParsedEvent):
        """Handle task tracker events."""
        op = event.data.get('operation')

        if op == TaskOperation.ADD.value:
            task = event.data.get('task', {})
            self.tasks[task.get('id')] = TaskInfo(
                task_id=task.get('id'),
                title=task.get('title', ''),
                description=task.get('description', ''),
                status=task.get('status', 'pending'),
                created_at=task.get('created_at')
            )

        elif op == TaskOperation.UPDATE.value:
            task_id = event.data.get('task_id')
            if task_id and task_id in self.tasks:
                self.tasks[task_id].status = event.data.get('status', 'unknown')
                if event.data.get('status') == 'completed':
                    self.tasks[task_id].completed_at = event.timestamp

    # ============== PUBLIC API ==============

    def get_chat_messages(self) -> List[ChatMessage]:
        """Get all parsed chat messages."""
        return self.chat_messages

    def get_current_status(self) -> Optional[Dict]:
        """Get current status bar state."""
        return self.current_status

    def get_tasks(self) -> List[TaskInfo]:
        """Get all tracked tasks."""
        return list(self.tasks.values())

    def is_complete(self) -> bool:
        """Check if response stream is complete."""
        return self.current_status and self.current_status.get('status') == 'idle'

    def get_final_content(self) -> str:
        """Get concatenated final text content."""
        contents = []
        for msg in self.chat_messages:
            contents.append(msg.content)
        return "\n\n".join(contents)

    def render_for_ui(self) -> Dict[str, Any]:
        """
        Render complete state for chat UI.
        Returns structured data ready for frontend rendering.
        """
        return {
            'status': {
                'is_processing': self.is_processing,
                'current_message': self.current_status.get('message') if self.current_status else None,
                'detail': self.current_status.get('detail') if self.current_status else None,
            },
            'tasks': [
                {
                    'id': t.task_id,
                    'title': t.title,
                    'description': t.description,
                    'status': t.status,
                    'is_done': t.status == 'completed'
                }
                for t in self.tasks.values()
            ],
            'messages': [
                {
                    'content': m.content,
                    'is_markdown': m.is_markdown,
                    'component_id': m.component_id,
                    'timestamp': m.timestamp
                }
                for m in self.chat_messages
            ],
            'is_complete': self.is_complete()
        }


# ============== USAGE EXAMPLE ==============

def example_usage():
    """Example of handling Vanna 2.0 stream in a chat UI."""

    handler = VannaStreamHandler()

    # Simulate streaming lines (normally from SSE/WebSocket)
    stream_lines = [
        # ... your stream data here ...
    ]

    for line in stream_lines:
        event = handler.parse_line(line)

        # Real-time UI updates
        if event.event_type == EventType.STATUS_BAR:
            # Update status indicator
            print(f"Status: {event.data.get('message')}")

        elif event.event_type == EventType.TASK_TRACKER:
            # Update task list
            print(f"Task update: {event.data.get('operation')}")

        elif event.event_type == EventType.TEXT:
            # Render message bubble
            print(f"New message: {event.data.get('content')[:50]}...")

    # Final render
    ui_state = handler.render_for_ui()
    print("\nFinal UI State:")
    print(json.dumps(ui_state, indent=2, default=str))


if __name__ == "__main__":
    # Test with provided data
    test_data = """PASTE_YOUR_STREAM_DATA_HERE"""

    handler = VannaStreamHandler()
    for line in test_data.strip().split("\n"):
        handler.parse_line(line)

    print("=== Final Content ===")
    print(handler.get_final_content())
    print("\n=== UI State ===")
    print(json.dumps(handler.render_for_ui(), indent=2, default=str))