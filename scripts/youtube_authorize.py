"""Run locally once to grant your own uploader access to the intended YouTube channel."""
import argparse
import json
import os
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from mezmur.upload import SCOPES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-secrets", required=True, type=Path, help="Downloaded Desktop app OAuth client JSON")
    parser.add_argument("--output", type=Path, default=Path(".runtime/youtube-oauth.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path instead of overwriting credentials")
    config = json.loads(args.client_secrets.read_text(encoding="utf-8"))
    if "installed" not in config:
        parser.error("Create a Desktop app OAuth client, not a Web application or service account")
    flow = InstalledAppFlow.from_client_config(config, SCOPES, autogenerate_code_verifier=True)
    credentials = flow.run_local_server(host="127.0.0.1", port=0, access_type="offline", prompt="consent")
    if not credentials.refresh_token:
        parser.error("Google returned no refresh token; revoke this application's grant and authorize again")
    service = build("youtube", "v3", credentials=credentials, cache_discovery=False)
    channels = service.channels().list(part="snippet", mine=True, maxResults=50).execute().get("items", [])
    if len(channels) != 1:
        parser.error("Authorize exactly one YouTube channel or Brand Account")
    data = {"client_id": credentials.client_id, "client_secret": credentials.client_secret,
            "refresh_token": credentials.refresh_token, "scopes": SCOPES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    print(f"Authorized channel: {channels[0]['snippet']['title']}")
    print(f"YOUTUBE_CHANNEL_ID: {channels[0]['id']}")
    print(f"Private credential file: {args.output}")
    print("Set repository secret YOUTUBE_OAUTH_JSON from this file. Never commit it or attach it to a workflow artifact.")


if __name__ == "__main__":
    main()
