from django.db import models

class Meeting(models.Model):
    # meeting_id INTEGER PK
    meeting_id = models.AutoField(primary_key=True)

    # host_id FK → accounts.User.user_id
    host = models.ForeignKey(
        "accounts.User",
        on_delete=models.PROTECT,
        related_name="hosted_meetings",
        null=True,
        blank=True,
    )

    transcript = models.TextField()
    title = models.CharField(max_length=90)
    meet_date_time = models.DateTimeField()
    place = models.CharField(max_length=90)

    summary = models.TextField(blank=True)
    meeting_notes = models.TextField(blank=True)
    record_url = models.ForeignKey(
        "meetings.S3File",
        on_delete=models.SET_NULL,
        to_field="record_url",
        null=True,
        blank=True,
        db_column="record_url",
    )
    domain = models.CharField(max_length=12, null=True, blank=True)
    private_yn = models.BooleanField(default=False)
    class Meta:
        db_table = "meeting_tbl"

    def __str__(self):
        return f"[{self.meet_id}] {self.title}"

class Attendee(models.Model):
    meeting = models.ForeignKey(
        Meeting,
        on_delete=models.CASCADE,
        related_name="attendees",
    )
    user = models.ForeignKey(
        "accounts.User",
        on_delete=models.CASCADE,
        related_name="attendances",
    )

    class Meta:
        db_table = "attendee_tbl"
        unique_together = ("meeting", "user")

    def __str__(self):
        return f"{self.meeting_id} - {self.user_id}"
    
class Task(models.Model):
    task_id = models.AutoField(primary_key=True)

    meeting = models.ForeignKey(
        Meeting,
        on_delete=models.CASCADE,
        related_name="tasks",
    )

    # assignee_id FK → User (nullable)
    assignee = models.ForeignKey(
        "accounts.User",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="tasks",
    )

    task_content = models.TextField(null=True, blank=True)
    due_date = models.DateField(null=True, blank=True)

    class Meta:
        db_table = "task_tbl"

    def __str__(self):
        return f"[{self.meeting_id}] {self.task_content}"

class S3File(models.Model):
    # s3_key (S3 객체 경로)
    s3_key = models.CharField(max_length=512, primary_key=True)
    original_name = models.CharField(max_length=255)
    record_url = models.URLField(max_length=512, unique=True)
    delete_at = models.DateTimeField()      # 삭제 예정 시각

    class Meta:
        db_table = "s3_file"

    def __str__(self):
        return self.s3_key