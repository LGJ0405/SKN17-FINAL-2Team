import requests
from django.conf import settings


url = f"https://{settings.POD_ID}-8000.proxy.runpod.net/"

def runpod_health():
    res = requests.get(url + "health") 
    return res.status_code

def get_stt(presigned_url):
    res = requests.post(
        url + "stt",
        json={'audio_url': presigned_url}
    )
    return res

def get_sllm(transcript, domain=""):
    """
    domain: string or list of strings. Normalized to list for API.
    """
    if isinstance(domain, (list, tuple)):
        domain_payload = list(domain)
    elif domain:
        domain_payload = [domain]
    else:
        domain_payload = []

    res = requests.post(
        url + "inference",
        json={
            'transcript': transcript,
            'domain': domain_payload
        }
    )
    return res
