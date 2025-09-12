import boto3
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

def index(request):
    return render(request, 'chat/index.html')

@csrf_exempt
def chat(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '')
        if not user_message:
            return JsonResponse({'error': 'No message provided'}, status=400)

        try:
            # Prepare the request for Titan model
            body = json.dumps({
                "inputText": user_message,
                "textGenerationConfig": {
                    "maxTokenCount": 100,
                    "temperature": 0.7,
                    "topP": 1
                }
            })

            # Invoke the model
            response = bedrock.invoke_model(
                modelId='amazon.titan-text-lite-v1',
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            # Parse the response
            response_body = json.loads(response['body'].read())
            bot_message = response_body['results'][0]['outputText']

            return JsonResponse({'response': bot_message})

        except Exception as e:
            error_str = str(e)
            if 'AccessDeniedException' in error_str:
                # Mock response for demo
                mock_response = f"This is a mock response to '{user_message}'. Please enable the Titan model in AWS Bedrock console to get real AI responses."
                return JsonResponse({'response': mock_response})
            else:
                return JsonResponse({'error': error_str}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)
