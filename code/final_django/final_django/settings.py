from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
GOOGLE_OAUTH2_CLIENT_SECRETS_JSON = BASE_DIR / "client_secret.json"
GOOGLE_OAUTH2_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
]
GOOGLE_OAUTH2_REDIRECT_URI = "http://localhost:8000/oauth2callback/"
# GOOGLE_OAUTH2_REDIRECT_URI = "https://malhaneundaero.com/oauth2callback/"

load_dotenv(os.path.join(BASE_DIR, '.env'))

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")

DEBUG = True
# DEBUG = False

ALLOWED_HOSTS = [
    '127.0.0.1',
    'localhost',
]

AWS_ELASTIC_IP = os.getenv("AWS_ELASTIC_IP")
if AWS_ELASTIC_IP:
    ALLOWED_HOSTS.append(AWS_ELASTIC_IP)

DOMAIN_URL = os.getenv("DOMAIN_URL")
if DOMAIN_URL:
    ALLOWED_HOSTS += [DOMAIN_URL, f"www.{DOMAIN_URL}"]

ALB_DNS_NAME = os.getenv("ALB_DNS_NAME")
if ALB_DNS_NAME:
    ALLOWED_HOSTS.append(ALB_DNS_NAME)

ALLOWED_HOSTS.append("172.31.37.73")

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    'storages',
    "core",
    "users",
    "meetings",

    'google_calendar',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
]

ROOT_URLCONF = 'final_django.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / "templates"],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                "meetings.views.today_meetings",
            ],
        },
    },
]

WSGI_APPLICATION = 'final_django.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.mysql',
#         'NAME': 'proddb',          # 실제 DB 이름으로 수정
#         'USER': 'admin',           # 실제 유저명으로 수정
#         'PASSWORD': os.getenv('DATABASES_PASSWORD'),
#         'HOST': os.getenv('DATABASES_HOST'),
#         'PORT': '3306',
#         'OPTIONS': {
#             'sql_mode': 'STRICT_TRANS_TABLES',
#             'charset': 'utf8mb4',
#         },
#     },
# }

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'ko-kr'
TIME_ZONE = 'Asia/Seoul'

USE_I18N = True
USE_TZ = False  # 시간대

# 세션 설정 (구글 OAuth 상태 유지를 위해)
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
SESSION_COOKIE_AGE = 1209600  # 2주
SESSION_SAVE_EVERY_REQUEST = True  # OAuth 리다이렉트 시 세션 유지
SESSION_COOKIE_HTTPONLY = False  # OAuth 리다이렉트를 위해 False
SESSION_COOKIE_SECURE = False  # HTTP localhost를 위해
SESSION_COOKIE_SAMESITE = None  # OAuth 리다이렉트를 위해 None (쿠키 전송 허용)
SESSION_COOKIE_NAME = 'sessionid'
SESSION_COOKIE_DOMAIN = None  # localhost에서 작동하도록

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [
    BASE_DIR / 'static',  
]

# 폰트 경로 (프로젝트 루트/static/font/malgun.ttf)
KOREAN_FONT_NAME = "MalgunGothic"
KOREAN_FONT_PATH = BASE_DIR / "static" / "font" / "malgun.ttf"

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# 파일은 S3에 저장
DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'

STORAGES = {
    "default": {
        "BACKEND": "storages.backends.s3boto3.S3Boto3Storage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")
AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME', 'ap-northeast-2')
POD_ID = os.getenv("POD_ID")


CSRF_TRUSTED_ORIGINS = [
    "http://" + os.getenv("AWS_ELASTIC_IP") + ":8080",
    "https://" + os.getenv("DOMAIN_URL")
]
