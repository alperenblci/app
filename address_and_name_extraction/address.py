import requests
import json
import urllib.parse
import time


def extract_address(image_description, name_address):
    print("Extracting the address")
    start = time.time()
    geocode_request = urllib.parse.urlencode({
        'auth': "839468063389088654876x108095",
        'scantext': image_description,
        'json': 1
    })

    geocode_response = requests.get(f"https://geocode.xyz/?{geocode_request}")
    geocode_response_json = json.loads(geocode_response.text)
    matches = geocode_response_json["match"]

    if matches:
        print(matches)
        for match in matches:
            if match["matchtype"] == "street" or match["matchtype"] == "locality":
                name_address[1] = f"{match['location']}, Latitude: {match['latt']} Longutide: {match['longt']}"
                end = time.time()
                print(
                    f"Extracted address. Address is {name_address[1]}, elapsed time is {end - start} seconds")
                return
