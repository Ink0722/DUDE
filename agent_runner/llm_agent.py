from src.model import build_backend
from src.parser import parse_action_call, extract_action, extract_final_answer, extract_thought
from .prompt_template import static_template

import os
from string import Template
from typing import List, Callable
import inspect

class ReActAgent:
    def __init__(
        self,
        tools: List[Callable],
        model: str,
        project_directory: str,
        device: str | None = None,
    ):
        
        """Generic ReAct agent.

        Args:
            tools: Tool functions available to the model.
            model: Model identifier, for example "Qwen/Qwen3-VL-4B-Instruct".
            project_directory: Project root used by the agent prompt.
            device: Optional runtime device, for example "cuda" or "cpu".
        """

        self.tools = {func.__name__: func for func in tools}
        self.model = model
        self.project_directory = project_directory
        self.device = device
        self.client = build_backend(model_name=self.model, device=self.device)
        self.experience = "To be update"

    def run(self, user_input: str|None=None, image_paths: List[str]|None=None, max_steps: int=3):
        messages = []
        step_count = 0
        # Build the system message first.
        messages.append({
            "role": "system",
            "content": self.render_system_prompt(static_template)
        })
        
        # Build the multimodal user message.
        user_content = []
        if image_paths is not None:
            for img_path in image_paths:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_path
                    }
                })
        if user_input is not None:
            user_content.append({
                "type": "text",
                "text": user_input
            })
        
        messages.append({
            "role": "user",
            "content": user_content
        })

        while True:

            # Stop once the agent reaches the configured step limit.
            if step_count >= max_steps:
                print(f"\n[Stop] Max steps {max_steps} reached.")
                return "Max steps reached"

            # Query the model.
            content = self.client.call_model(messages)
            step_count += 1

            # Parse the <thought> section when present.
            thought = extract_thought(content)
            if thought is not None:
                print(f"Thought: {thought}")

            # Return immediately when the model emits a final answer.
            final_answer = extract_final_answer(content)
            if final_answer is not None:
                if "</final_answer>" not in content:
                    print("[Error]: solely <final_answer> without </>")
                return final_answer

            # Parse the <action> section.
            action = extract_action(content)
            if action is None:
                print("[Error]: Unmatch")
                print("[UNmatch]:",content)
                continue

            tool_name, args, kwargs = parse_action_call(action)
            print(f"Parameter Phase Action: {tool_name}")

            # Format positional and keyword arguments for readable logging.
            pos_str = ", ".join(repr(a) for a in args)
            kw_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            all_str = ", ".join(s for s in [pos_str, kw_str] if s)  # Avoid dangling commas when kwargs are empty.
            print(f"Action: {tool_name}({all_str})")
            # Ask for confirmation before executing terminal commands.
            should_continue = input("\n\nContinue? (Y/N): ") if tool_name == "run_terminal_command" else "y"
            if should_continue.lower() != 'y':
                print("\n\nOperation canceled.")
                return "Operation canceled by user."

            try:
                observation = self.tools[tool_name](*args, **kwargs)
            except Exception as e:
                observation = f"Tool execution error: {str(e)}"
            print(f"Observation: {observation}")
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})

    def get_tool_list(self) -> str:
        """Return one formatted line of documentation for each registered tool."""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)

    def render_system_prompt(self, system_prompt_template: str) -> str:
        """Render the system prompt template with runtime variables."""
        tool_list = self.get_tool_list()
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_directory, f))
            for f in os.listdir(self.project_directory)
        )
        return Template(system_prompt_template).substitute(
            # operating_system=self.get_operating_system_name(),
            tool_list=tool_list,
            experience=self.experience
        )

    def call_model(self, messages):
        print("\n\nCalling model...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        return content




