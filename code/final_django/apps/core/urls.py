from django.urls import path
from .views import (
    HomeView,
    AdminHomeView,
    admin_member_create,
    admin_member_update,
    admin_member_delete,
    admin_member_reset_password
)

app_name = "core"

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("admin/", AdminHomeView.as_view(), name="admin_home"),
    path("admin/member/create/", admin_member_create, name="admin_member_create"),
    path("admin/member/<str:user_id>/update/", admin_member_update, name="admin_member_update"),
    path("admin/member/<str:user_id>/delete/", admin_member_delete, name="admin_member_delete"),
    path("admin/member/<str:user_id>/reset-password/", admin_member_reset_password, name="admin_member_reset_password"),
]
