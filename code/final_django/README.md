# FINAL_DJANGO

# 배포 가이드

# 1. EC2 SSH 접속
```bash
ssh -i <pem키 경로> ubuntu@<EC2 IP>
```
# 2. 필수 패키지 설치
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
```
# 3. Docker GPG 키 등록
```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```
# 4. Docker 공식 저장소 추가
```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
# 5. Docker & Compose 플러그인 설치
```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
# 6. Docker 명령어 등록
```bash
sudo usermod -aG docker ubuntu
newgrp docker
```
# 7. git clone
```bash
git clone https://github.com/skn17-final-2team/final_django.git
cd final_django
```
# 8 .env 설정 변경
```
AWS_ELASTIC_IP=<EC2 IP>
POD_ID=<POD ID>
```
# 9. .env 파일 EC2로 옮기기
```bash
scp -i <pem키 경로> <.env 경로> ubuntu@[EC2 IP]:/home/ubuntu/final_django
```
# 10. settings DEBUG = False 주석 풀기
```bash
vim final_django/settings.py
```
# 11. Docker 빌드 및  실행
```bash
docker compose build --no-cache
docker compose up -d
```
# 12. Docker 다운
```bash
docker compose down
```