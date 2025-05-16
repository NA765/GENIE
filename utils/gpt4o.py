import os
import io
import time
import base64
from mimetypes import guess_type
import openai
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from PIL import Image

from typing import List, Optional, Union



OPENAI_API_KEY = "YOUR API KEY HERE"



def local_image_to_data_url(image_source, image_format=None):
    """
    Convert a local image path or PIL.Image object to a Data URL
    
    :param image_source: Image file path (str) or PIL.Image object
    :param image_format: Format specification when input is PIL.Image (e.g., "PNG", "JPEG")
    :return: Data URL string
    """
    if isinstance(image_source, str):
        # Handle file path input
        mime_type, _ = guess_type(image_source)
        mime_type = mime_type or 'application/octet-stream'  # Default MIME type if none is found
        
        with open(image_source, "rb") as image_file:
            image_data = image_file.read()
    
    elif isinstance(image_source, Image.Image):
        # Handle PIL.Image object input
        image_format = image_format or image_source.format or 'PNG'
        image_buffer = io.BytesIO()
        
        # Save image to in-memory buffer
        image_source.save(image_buffer, format=image_format)
        image_data = image_buffer.getvalue()
        
        # Set MIME type based on format
        if image_format.upper() == 'JPEG':
            mime_type = 'image/jpeg'
        else:
            mime_type = f'image/{image_format.lower()}'
    
    else:
        raise ValueError("Invalid input type. Expected file path (str) or PIL.Image object")
    
    # Base64 encoding and URL construction
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_data}"




def gpt4o_response(
        prompt: Union[str, List[str]], 
        image_path: Union[Optional[str], List[str]] = None, 
        model_version: str = "chatgpt-4o-latest",
        max_retry: int = 15, 
        max_tokens: int = 2000,
        ):

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    retry_count = 0
    response = None
        
    prompts = [prompt] if isinstance(prompt, str) else prompt
    image_paths = [image_path] if isinstance(image_path, str) else image_path

    message = list()
    if image_paths:
        for image_path in image_paths:
            try:
                data_url = local_image_to_data_url(image_path)
            except FileNotFoundError:
                data_url = image_path
            message.append(
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": f"{data_url}"
                    }
                }
            )
            
    for prompt in prompts:
        message.append(
            { 
                "type": "text", 
                "text": prompt 
            }
        )


    message_body = [
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": message } 
            ]

    while retry_count < max_retry:
        try:
            response = client.chat.completions.create(
                model=model_version,
                messages=message_body,
                max_tokens=max_tokens
            )
            print(response.model)
            response = response.choices[0].message.content
            break
        
        except openai.BadRequestError: # policy voilation content generated
            retry_count += 5
            print('Incorrect request format or policy voilation content detected, trying to retry for the %dth time' % (retry_count//5))

        except openai.RateLimitError: # request too often
            time.sleep(1)
            retry_count += 1
            print('Request too often, trying to retry for the %dth time' % retry_count)

        except openai.APITimeoutError: # request timed out
            time.sleep(1)
            retry_count += 1
            print('Request timed out, trying to retry for the %dth time' % retry_count)

        except openai.InternalServerError:
            time.sleep(1)
            print('The server had an error processing request. Request of %dth time will be resent.' % retry_count)
    
    if not response:
        print('Failed to get respone after %d times retries' % max_retry)

    return response




# get response from gpt-4o based on prompt and image given
# you can use this version of gpt-4o if Azure OpenAI is available 
def gpt4o_response_legacy(
        prompt: Union[str, List[str]], 
        image_path: Union[Optional[str], Image.Image, List[str], List[Image.Image]] = None, 
        endpoint_url = "https://mcg-openai-swedencentral-b.openai.azure.com/",
        deployment_name = "gpt-4o",
        max_retry: int = 15, 
        max_tokens: int = 2000,
        verbose: bool = True,
        ):

    endpoint = os.getenv("ENDPOINT_URL", endpoint_url)
    deployment = os.getenv("DEPLOYMENT_NAME", deployment_name)

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default")
        
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-11-20-preview",
    )

    retry_count = 0
    response = None
        
    prompts = [prompt] if isinstance(prompt, str) else prompt
    image_paths = [image_path] if (isinstance(image_path, str) or isinstance(image_path, Image.Image)) else image_path

    message = list()
    for prompt in prompts:
        message.append(
            { 
                "type": "text", 
                "text": prompt 
            }
        )

    if image_paths:
        for image_path in image_paths:
            try:
                data_url = local_image_to_data_url(image_path)
            except FileNotFoundError:
                data_url = image_path
            message.append(
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": f"{data_url}"
                    }
                }
            )
    message_body = [
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": message } 
    ]


    while retry_count < max_retry:
        try:
            
            response = client.chat.completions.create(
                model=deployment,
                messages=message_body,
                max_tokens=max_tokens
            )
            # print(response.model)
            response = response.choices[0].message.content
            break
        
        except openai.BadRequestError: # policy voilation content generated
            retry_count += 5
            if verbose:
                print('Incorrect request format or policy voilation content detected, trying to retry for the %dth time' % (retry_count//5))

        except openai.RateLimitError: # request too often
            time.sleep(1)
            retry_count += 1
            if verbose:
                print('Request too often, trying to retry for the %dth time' % retry_count)

        except openai.APITimeoutError: # request timed out
            time.sleep(1)
            retry_count += 1
            if verbose:
                print('Request timed out, trying to retry for the %dth time' % retry_count)

        except openai.InternalServerError:
            time.sleep(1)
            if verbose:
                print('The server had an error processing request. Request of %dth time will be resent.' % retry_count)
    
    if not response:
        if verbose:
            print('Failed to get respone after %d times retries' % max_retry)

    return response