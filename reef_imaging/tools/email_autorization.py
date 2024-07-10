import os
import json

# This class is used to check if the user is authorized to use the chatbot
class AuthenticatedUser:
    def __init__(self, login_required=True):
        self.login_required = login_required
        self.authorized_emails = self.load_authorized_emails()
        print(f"Authorized emails: {self.authorized_emails}")

    def load_authorized_emails(self, login_required=True):
        if login_required:
            authorized_users_path = os.environ.get("BIOIMAGEIO_AUTHORIZED_USERS_PATH")
            if authorized_users_path:
                assert os.path.exists(
                    authorized_users_path
                ), f"The authorized users file is not found at {authorized_users_path}"
                with open(authorized_users_path, "r") as f:
                    authorized_users = json.load(f)["users"]
                authorized_emails = [
                    user["email"] for user in authorized_users if "email" in user
                ]
            else:
                authorized_emails = None
        else:
            authorized_emails = None
        return authorized_emails



    def check_permission(self, user):
        if user['is_anonymous']:
            return False
        if self.authorized_emails is None or user["email"] in self.authorized_emails:
            return True
        else:
            return False

    async def ping(self, context=None):
        if self.login_required and context and context.get("user"):
            assert self.check_permission(
                context.get("user")
            ), "You don't have permission to use the chatbot, please sign up and wait for approval"
        return "pong"
