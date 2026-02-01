import os
import random
import json
from enum import Enum

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

g_tools = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_your_try",
            "description": "Use this function to evaluate your guessed number. Returns a message indicating whether your guess was too high, too low, or correct.",
            "parameters": {
                "type": "object",
                "properties": {
                    "guess": {
                        "type": "number",
                        "description": "Your guess number.",
                    }
                },
                "required": ["guess"],
            },
        },
    },
]

# Gets a random number between 1 and 100
app_random_number = random.randint(1, 100)

TOO_HIGH_MESSAGE = "Too high!"
TOO_LOW_MESSAGE = "Too low!"
THAT_IS_RIGHT_MESSAGE = "That's right!"

def evaluate_your_try(guess):
    """
    Evaluate the guess made by the user.

    :param guess: The guess made by the LLM
    :return: the evaluation message
    """
    print(f"LLM: {guess}")
    if guess > app_random_number:
        return TOO_HIGH_MESSAGE
    elif guess < app_random_number:
        return TOO_LOW_MESSAGE
    return THAT_IS_RIGHT_MESSAGE

available_functions = {
    "evaluate_your_try": evaluate_your_try,
}

class GameState(Enum):
    INIT = 0
    PLAYING = 1
    END = 2

class OpenAIGuessGameAgent:
    def __init__(self, client, model = "gpt-5-nano", max_tries=5):
        self.client = client
        self.model = model
        self.max_tries = max_tries
        self.state = GameState.INIT
        self.messages = []
        self.init()

    def init(self):
        """Initialize the game. Set up the initial messages."""
        start_game_message = f"ME: I am thinking of a number between 1 and 100 and you will try to guess it. You have {self.max_tries} tries to guess."
        print(start_game_message)

        self.messages = [
            {"role": "system", "content": "You are a fortune teller"},
            {"role": "user", "content": start_game_message},
        ]

    def play(self):
        """Play the game. The LLM will make guesses and the user will provide feedback."""
        self.state = GameState.PLAYING
        llm_try = 0

        try:
            while llm_try < self.max_tries:
                llm_try += 1
                self.call_llm(g_tools)
                if self.state == GameState.END:
                    return
        except Exception as e:
            print(f"Error: {e}")


        self.state = GameState.END
        final_message = f"ME: You lost! My number was {app_random_number}. How is it possible when you are the fortune teller?"
        print(final_message)
        self.messages.append({ "role": "user", "content": final_message })
        self.call_llm([])

    #
    def call_llm(self, tools):
        """
        Calls the LLM with the current messages and tools.

        :param tools: The available tools for the LLM
        :return: True if successful, False otherwise
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=tools
        )
        response_message = response.choices[0].message

        if response_message.tool_calls:
            return self.process_tool_response(response_message)
        return self.process_message(response_message)

    def process_tool_response(self, response_message):
        """
        Process the tool response from the LLM.

        :param response_message: The response message from the LLM
        :return: True if successful, False otherwise
        """
        self.messages.append(
            {
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response_message.tool_calls
                ],
            }
        )

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            tool_id = tool_call.id

            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)
            if function_response == THAT_IS_RIGHT_MESSAGE:
                self.state = GameState.END
            print(f"ME: {function_response}")

            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
            )

    def process_message(self, response_message):
        """
        Process the regular message from the LLM.

        :param response_message: The response message from the LLM
        :return: True if successful, False if it is called during the PLAYING state
        """
        final_content = response_message.content
        print(f"LLM: {final_content}")

        if self.state == GameState.PLAYING:
            raise Exception("Unexpected message from LLM during PLAYING state.")

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
    except Exception as e:
        print(f"Not connected to OpenAI: {e}")
        return

    agent = OpenAIGuessGameAgent(client, max_tries=5)
    agent.play()

if __name__ == '__main__':
    main()