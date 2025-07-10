import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_token_from_node(node_api_url, identity, room_name, role="admin", admin_message=""):
    url = f"{node_api_url}/token"
    headers = {"Content-Type": "application/json"}
    payload = {
        "identity": identity,
        "roomName": room_name,
        "role": role,
        "adminWelcomeMessage": admin_message
    }

    try:
        logger.info(f"Requesting token from: {url}")
        logger.info(f"Payload: {payload}")
        
        response = requests.post(url, json=payload, headers=headers)
        
        # Log response details
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        
        # Check if the response is empty
        if not response.text:
            logger.error("Received empty response from server")
            return None
            
        # Log the raw response for debugging
        logger.info(f"Raw response text: {response.text[:200]}...")  # First 200 chars
        
        response.raise_for_status()
        
        try:
            response_data = response.json()
            token = response_data.get("token")
            if not token:
                logger.error(f"Token not found in response. Response data: {response_data}")
                raise ValueError("Token not found in response")
            logger.info("Token successfully retrieved")
            return token
        except ValueError as json_err:
            logger.error(f"Failed to parse JSON response: {json_err}")
            logger.error(f"Response content: {response.text}")
            raise ValueError(f"Invalid JSON response: {json_err}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None