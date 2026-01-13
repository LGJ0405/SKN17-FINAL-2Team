from django.db import models
from users.models import User


class GoogleCalendarToken(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="google_calendar_token",
    )
    token_json = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "google_calendar_token"

    def __str__(self):
        return f"GoogleToken({self.user.user_id})"


class OAuthState(models.Model):
    """OAuth 인증 과정에서 state와 user_id를 임시 저장"""
    state = models.CharField(max_length=255, unique=True, primary_key=True)
    user_id = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "oauth_state"
