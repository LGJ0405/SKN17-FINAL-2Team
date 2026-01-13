from django.urls import path
from .views import (
    MeetingCreateView,
    MeetingListAllView,
    MeetingListMineView,
    MeetingListDeptView,
    MeetingRecordView,
    MeetingTranscriptView,
    MeetingDetailView,
    meeting_record_upload,
    meeting_audio_download,
    meeting_record_url_set,
    meeting_summary,
    MeetingSttRenderingView,
    MeetingSllmRenderingView,
    meeting_transcript_prepare,
    meeting_sllm_prepare,
    meeting_transcript_save,
    minutes_download,
    minutes_save,
    meeting_transcript_api,
    tasks_save,
)

app_name = "meetings"

urlpatterns = [
    path("new/", MeetingCreateView.as_view(), name="meeting_create"),
    path("<int:meeting_id>/record/", MeetingRecordView.as_view(), name="meeting_record"),
    path("<int:meeting_id>/upload/", meeting_record_upload, name="meeting_upload"),
    path("<int:meeting_id>/record_url/set/", meeting_record_url_set, name="meeting_record_url_set"),

    path("<int:meeting_id>/rendering/stt/", MeetingSttRenderingView.as_view(), name="rendering_stt"),
    path("<int:meeting_id>/rendering/sllm/", MeetingSllmRenderingView.as_view(), name="rendering_sllm"),
    path("<int:meeting_id>/transcript/prepare/", meeting_transcript_prepare, name="meeting_transcript_prepare"),
    path("<int:meeting_id>/sllm/prepare/", meeting_sllm_prepare, name="meeting_sllm_prepare"),

    path("<int:meeting_id>/transcript/", MeetingTranscriptView.as_view(), name="meeting_transcript"),
    path("<int:meeting_id>/transcript_api/", meeting_transcript_api, name="meeting_transcript_api"),
    path("<int:meeting_id>/transcript/save/", meeting_transcript_save, name="meeting_transcript_save"),

    path("<int:meeting_id>/detail", MeetingDetailView.as_view(), name="meeting_detail"),
    path("<int:meeting_id>/audio/download/", meeting_audio_download, name="audio_download"),

    path("<int:meeting_id>/minutes/save/", minutes_save, name="minutes_save"),
    path("<int:meeting_id>/minutes/download/<str:fmt>/", minutes_download, name="minutes_download"),
    path("<int:meeting_id>/tasks/save/", tasks_save, name="tasks_save"),

    path("list/all/", MeetingListAllView.as_view(), name="list_all"),
    path("list/mine/", MeetingListMineView.as_view(), name="list_mine"),
    path("list/dept/", MeetingListDeptView.as_view(), name="list_dept"),
]
