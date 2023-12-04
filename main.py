# https://github.com/openai/openai-cookbook/blob/main/examples/Fine_tuning_for_function_calling.ipynb


import os
import sys
import time
import ast
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Generator
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
import numpy as np

OPEN_AI_CLIENT = OpenAI()


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
def get_chat_completion(
    messages: list[dict[str, str]],
    # model: str = "gpt-3.5-turbo-1106",
    model: str = "gpt-4-1106-preview",
    max_tokens=500,
    temperature=1.0,
    stop=None,
    functions=None,
    response_format=None,
) -> str:
    params = {
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stop': stop,
    }
    if response_format:
        params['response_format'] = response_format
    if functions:
        params['functions'] = functions

    completion = OPEN_AI_CLIENT.chat.completions.create(**params)
    return completion.choices[0].message


DRONE_SYSTEM_PROMPT = """You are an intelligent AI that controls a drone. Given a command or request from the user,
call one of your functions to complete the request. If the request cannot be completed by your available functions, call the reject_request function.
If the request is ambiguous or unclear, reject the request."""


function_list = [
    {
        "name": "takeoff_drone",
        "description": "Initiate the drone's takeoff sequence.",
        "parameters": {
            "type": "object",
            "properties": {
                "altitude": {
                    "type": "integer",
                    "description": "Specifies the altitude in meters to which the drone should ascend."
                }
            },
            "required": ["altitude"]
        }
    },
    {
        "name": "land_drone",
        "description": "Land the drone at its current location or a specified landing point.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "enum": ["current", "home_base", "custom"],
                    "description": "Specifies the landing location for the drone."
                },
                "coordinates": {
                    "type": "object",
                    "description": "GPS coordinates for custom landing location. Required if location is 'custom'."
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "control_drone_movement",
        "description": "Direct the drone's movement in a specific direction.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward", "left", "right", "up", "down"],
                    "description": "Direction in which the drone should move."
                },
                "distance": {
                    "type": "integer",
                    "description": "Distance in meters the drone should travel in the specified direction."
                }
            },
            "required": ["direction", "distance"]
        }
    },
    {
        "name": "set_drone_speed",
        "description": "Adjust the speed of the drone.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed": {
                    "type": "integer",
                    "description": "Specifies the speed in km/h."
                }
            },
            "required": ["speed"]
        }
    },
    {
        "name": "control_camera",
        "description": "Control the drone's camera to capture images or videos.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["photo", "video", "panorama"],
                    "description": "Camera mode to capture content."
                },
                "duration": {
                    "type": "integer",
                    "description": "Duration in seconds for video capture. Required if mode is 'video'."
                }
            },
            "required": ["mode"]
        }
    },
    {
        "name": "control_gimbal",
        "description": "Adjust the drone's gimbal for camera stabilization and direction.",
        "parameters": {
            "type": "object",
            "properties": {
                "tilt": {
                    "type": "integer",
                    "description": "Tilt angle for the gimbal in degrees."
                },
                "pan": {
                    "type": "integer",
                    "description": "Pan angle for the gimbal in degrees."
                }
            },
            "required": ["tilt", "pan"]
        }
    },
    {
        "name": "set_drone_lighting",
        "description": "Control the drone's lighting for visibility and signaling.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["on", "off", "blink", "sos"],
                    "description": "Lighting mode for the drone."
                }
            },
            "required": ["mode"]
        }
    },
    {
        "name": "return_to_home",
        "description": "Command the drone to return to its home or launch location.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "set_battery_saver_mode",
        "description": "Toggle battery saver mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["on", "off"],
                    "description": "Toggle battery saver mode."
                }
            },
            "required": ["status"]
        }
    },
    {
        "name": "set_obstacle_avoidance",
        "description": "Configure obstacle avoidance settings.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["on", "off"],
                    "description": "Toggle obstacle avoidance."
                }
            },
            "required": ["mode"]
        }
    },
    {
        "name": "set_follow_me_mode",
        "description": "Enable or disable 'follow me' mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["on", "off"],
                    "description": "Toggle 'follow me' mode."
                }
            },
            "required": ["status"]
        }
    },
    {
        "name": "calibrate_sensors",
        "description": "Initiate calibration sequence for drone's sensors.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "set_autopilot",
        "description": "Enable or disable autopilot mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["on", "off"],
                    "description": "Toggle autopilot mode."
                }
            },
            "required": ["status"]
        }
    },
    {
        "name": "configure_led_display",
        "description": "Configure the drone's LED display pattern and colors.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "enum": ["solid", "blink", "pulse", "rainbow"],
                    "description": "Pattern for the LED display."
                },
                "color": {
                    "type": "string",
                    "enum": ["red", "blue", "green", "yellow", "white"],
                    "description": "Color for the LED display. Not required if pattern is 'rainbow'."
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "set_home_location",
        "description": "Set or change the home location for the drone.",
        "parameters": {
            "type": "object",
            "properties": {
                "coordinates": {
                    "type": "object",
                    "description": "GPS coordinates for the home location."
                }
            },
            "required": ["coordinates"]
        }
    },
    {
        "name": "reject_request",
        "description": "Use this function if the request is not possible.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
]


# # test 1
straightforward_prompts = ['Land the drone at the home base',
                           'Take off the drone to 50 meters',
                           'change speed to 15 kilometers per hour',
                           'turn into an elephant!']
# for prompt in straightforward_prompts:
#   messages = []
#   messages.append({"role": "system", "content": DRONE_SYSTEM_PROMPT})
#   messages.append({"role": "user", "content": prompt})
#   completion = get_chat_completion(
#       model="gpt-3.5-turbo-1106", messages=messages, functions=function_list)
#   print(prompt)
#   print(completion.function_call, '\n')


# # test 2
challenging_prompts = ['Play pre-recorded audio message',
                       'Initiate live-streaming on social media',
                       'Scan environment for heat signatures',
                       'Enable stealth mode',
                       "Change drone's paint job color"]
# for prompt in challenging_prompts:
# messages = []
# messages.append({"role": "system", "content": DRONE_SYSTEM_PROMPT})
# messages.append({"role": "user", "content": prompt})
# completion = get_chat_completion(
#     model="gpt-3.5-turbo-1106", messages=messages, functions=function_list)
# print(prompt)
# try:
#     print(completion.function_call)
#     print('\n')
# except:
#     print(completion.content)
#     print('\n')


# Generating synthetic data

placeholder_int = 'fill_in_int'
placeholder_string = 'fill_in_string'


def generate_permutations(params: Dict[str, Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
    """
    Generates all possible permutations for given parameters.

    :param params: Parameter dictionary containing required and optional fields.
    :return: A generator yielding each permutation.
    """

    # Extract the required fields from the parameters
    required_fields = params.get('required', [])

    # Generate permutations for required fields
    required_permutations = generate_required_permutations(
        params, required_fields)

    # Generate optional permutations based on each required permutation
    for required_perm in required_permutations:
        yield from generate_optional_permutations(params, required_perm)


def generate_required_permutations(params: Dict[str, Dict[str, Any]], required_fields: List[str]) -> List[Dict[str, Any]]:
    """
    Generates permutations for the required fields.

    :param params: Parameter dictionary.
    :param required_fields: List of required fields.
    :return: A list of permutations for required fields.
    """

    # Get all possible values for each required field
    required_values = [get_possible_values(
        params, field) for field in required_fields]

    # Generate permutations from possible values
    return [dict(zip(required_fields, values)) for values in itertools.product(*required_values)]


def generate_optional_permutations(params: Dict[str, Dict[str, Any]], base_perm: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Generates permutations for optional fields based on a base permutation.

    :param params: Parameter dictionary.
    :param base_perm: Base permutation dictionary.
    :return: A generator yielding each permutation for optional fields.
    """

    # Determine the fields that are optional by subtracting the base permutation's fields from all properties
    optional_fields = set(params['properties']) - set(base_perm)

    # Iterate through all combinations of optional fields
    for field_subset in itertools.chain.from_iterable(itertools.combinations(optional_fields, r) for r in range(len(optional_fields) + 1)):

        # Generate product of possible values for the current subset of fields
        for values in itertools.product(*(get_possible_values(params, field) for field in field_subset)):

            # Create a new permutation by combining base permutation and current field values
            new_perm = {**base_perm, **dict(zip(field_subset, values))}

            yield new_perm


def get_possible_values(params: Dict[str, Dict[str, Any]], field: str) -> List[Any]:
    """
    Retrieves possible values for a given field.

    :param params: Parameter dictionary.
    :param field: The field for which to get possible values.
    :return: A list of possible values.
    """

    # Extract field information from the parameters
    field_info = params['properties'][field]

    # Based on the field's type or presence of 'enum', determine and return the possible values
    if 'enum' in field_info:
        return field_info['enum']
    elif field_info['type'] == 'integer':
        return [placeholder_int]
    elif field_info['type'] == 'string':
        return [placeholder_string]
    elif field_info['type'] == 'boolean':
        return [True, False]
    elif field_info['type'] == 'array' and 'enum' in field_info['items']:
        enum_values = field_info['items']['enum']
        all_combinations = [list(combo) for i in range(
            1, len(enum_values) + 1) for combo in itertools.combinations(enum_values, i)]
        return all_combinations
    return []


INVOCATION_FILLER_PROMPT = """
1) Input reasonable values for 'fill_in_string' and 'fill_in_int' in the invocation here: {invocation}. Reasonable values are determined by the function definition. Use the
the entire function provided here :{function} to get context over what proper fill_in_string and fill_in_int values would be.

Example JSON:

Input: invocation: {{
    "name": "control_camera",
    "arguments": {{
      "mode":"video",
      "duration":"fill_in_int"
    }}
}},
function:{function}

Output: invocation: {{
    "name": "control_camera",
    "arguments": {{
      "mode":"video",
      "duration": 30
    }}
}}


MAKE SURE output is just a dictionary with keys 'name' and 'arguments', no other text or response.

Input: {invocation}
Output:
"""


COMMAND_GENERATION_PROMPT = """
You are to output 2 commands, questions or statements that would generate the inputted function and parameters.
Please make the commands or questions natural, as a person would ask, and the command or questions should be varied and not repetitive.
It should not always mirror the exact technical terminology used in the function and parameters, rather reflect a conversational and intuitive request.
For instance, the prompt should not be 'turn on the dome light', as that is too technical, but rather 'turn on the inside lights'.
Another example, is the prompt should not be 'turn on the HVAC', but rather 'turn on the air conditioning'. Use language a normal driver would use, even if
it is technically incorrect but colloquially used.

RULES: ALWAYS put a backwards slash before an apostrophe or single quote '. For example, do not say don't but say don\'t.
Prompts MUST be in double quotes as well.

Example JSON:

Input: {{'name': 'calibrate_sensors','arguments': {{}}'' }}
Prompt: ["The sensors are out of whack, can you reset them", "The calibration of the drone is off, fix it please!"]

Input: {{'name': 'set_autopilot','arguments': {{'status': 'off'}}}}
Prompt: ["OK, I want to take back pilot control now","Turn off the automatic pilot I'm ready control it"]

Input: {invocation}
Prompt:
"""


def create_commands(invocation_list):
    example_list = []
    for i, invocation in enumerate(invocation_list):
        print(
            f'\033[34m{np.round(100*i/len(invocation_list),1)}% complete\033[0m')
        print(invocation)
        request_prompt = COMMAND_GENERATION_PROMPT.format(
            invocation=invocation)

        print("Generating commands...")
        print(request_prompt)
        messages = [{"role": "user", "content": f"{request_prompt}"}]
        completion = get_chat_completion(
            messages, temperature=0.8, response_format={"type": "json_object"})
        command_dict = {
            "Input": invocation,
            "Prompt": completion
        }
        example_list.append(command_dict)
    return example_list


def train():
    input_objects = []
    all_but_reject = [f for f in function_list if f.get(
        'name') != 'reject_request']

    print("Generating training data...")
    print("Input objects all_but_reject function list")
    for function in all_but_reject:
        func_name = function["name"]
        params = function["parameters"]
        print("Generating permutations")
        for arguments in generate_permutations(params):
            if any(val in arguments.values() for val in ['fill_in_int', 'fill_in_str']):
                input_object = {
                    "name": func_name,
                    "arguments": arguments
                }

                content = INVOCATION_FILLER_PROMPT.format(
                    invocation=input_object, function=function)
                print("Generating prompts...")
                print(content)
                messages = [{"role": "user", "content": content}]
                input_object = get_chat_completion(
                    # model='gpt-3.5-turbo-1106', messages=messages, max_tokens=200, temperature=.1, response_format={"type": "json_object"}).content
                    model='gpt-4-1106-preview', messages=messages, max_tokens=200, temperature=.1, response_format={"type": "json_object"}).content
            else:
                input_object = {
                    "name": func_name,
                    "arguments": arguments
                }
            input_objects.append(input_object)

    training_examples_unformatted = create_commands(input_objects)
    training_examples = []
    for prompt in training_examples_unformatted:
        # adjust formatting for training data specs
        try:
            # prompt["Input"] = json.dumps(prompt["Input"])
            prompt["Input"] = ast.literal_eval(prompt["Input"])
        except:
            continue
        prompt['Input']['arguments'] = json.dumps(prompt['Input']['arguments'])
        content = prompt['Prompt'].content
        print("Training example:")
        print(content)
        print(prompt['Input'])
        training_examples.append({"messages": [{"role": "system", "content": DRONE_SYSTEM_PROMPT}, {"role": "user", "content": content}, {"role": "assistant", "function_call": prompt['Input']}],
                                  "functions": function_list})

    reject_list = ['Translate broadcast message to another language',
                   'Automatically capture photos when face is detected',
                   'Detect nearby drones',
                   'Measure wind resistance',
                   'Capture slow motion video',
                   "Adjust drone's altitude to ground level changes",
                   'Display custom message on LED display',
                   "Sync drone's time with smartphone",
                   'Alert when drone travels out of designated area',
                   'Detect moisture levels',
                   'Automatically follow GPS tagged object',
                   'Toggle night vision mode',
                   'Maintain current altitude when battery is low',
                   'Decide best landing spot using AI',
                   "Program drone's route based on wind direction"]

    reject_training_list = []
    for prompt in reject_list:
        # Adjust formatting
        print("Reject training example:")
        print(prompt)
        reject_training_list.append({"messages": [{"role": "system", "content": DRONE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}, {"role": "assistant", "function_call": {"name": "reject_request", "arguments": "{}"}}],
                                    "functions": function_list})

    training_list_total = training_examples+reject_training_list

    training_file = 'data/drone_training.jsonl'
    with open(training_file, 'w') as f:
        for item in training_list_total:
            json_str = json.dumps(item)
            f.write(f'{json_str}\n')
        f.close()
    return training_file


def do_finetine(training_file):
    file = OPEN_AI_CLIENT.files.create(file=Path(str(training_file)),
                                       purpose="fine-tune")
    file_id = file.id
    print(file_id)
    # create fine tuning job
    ft = OPEN_AI_CLIENT.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-3.5-turbo-1106",
        hyperparameters={
            "n_epochs": 2
        }
    )

    def get_finetune_job(job_id):
        job = OPEN_AI_CLIENT.fine_tuning.jobs.retrieve(job_id)
        print(job)
        return job

    while get_finetune_job(ft.id).status not in ['succeeded', 'failed', 'cancelled']:
        job = get_finetune_job(ft.id)
        print("Waiting: ", job.status)
        print(job)
        time.sleep(2)


def main():
    if len(sys.argv) < 2:
        print("Usage: python fine_tune_function_calling.py <command>")
        sys.exit(1)

    cmd = sys.argv[1]
    print(cmd)

    if cmd == 'train':
        cmd = train()
        if not os.path.exists(cmd):
            print(f"training file {cmd} does not exist")
            sys.exit(1)

        # training file must be json
        if not cmd.endswith(".jsonl"):
            print(f"training file {cmd} must be json")
            sys.exit(1)

        print("Training file created")
        print(
            f"use `python fine_tune_function_calling.py {cmd}` create a fine tuned model")
        return

    if cmd == 'test':
        for eval_question in straightforward_prompts + challenging_prompts:
            messages = []
            messages.append({"role": "system", "content": DRONE_SYSTEM_PROMPT})
            print("Testing...")
            print(eval_question)
            messages.append({"role": "user", "content": eval_question})
            if len(sys.argv) == 3:
                model = sys.argv[2]
            else:
                raise Exception("Please provide a model")
            completion = get_chat_completion(
                model=model, messages=messages, functions=function_list)
            print(completion.function_call, '\n')
        print("Testing complete")
        return

    if cmd:
        if not os.path.exists(cmd):
            print(f"Training file {cmd} does not exist")
            sys.exit(1)

        if not cmd.endswith(".jsonl"):
            print(f"Training file {cmd} must be jsonl")
            sys.exit(1)
        do_finetine(cmd)
        print("Fine tuning complete")
        return


if __name__ == "__main__":
    main()
