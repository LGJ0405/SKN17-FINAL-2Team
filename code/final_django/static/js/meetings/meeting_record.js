// ===== DOM 요소 =====
const dropzone = document.getElementById('dropzone');
const hiddenInput = document.getElementById('hidden-file');
const preview = document.getElementById('preview');
const uploadBtn = document.getElementById('upload-btn');
const csrftoken = document.querySelector("input[name='csrfmiddlewaretoken']")?.value;

// 녹음 관련 요소
const btnRecord = document.getElementById('btn-record');
const btnPause = document.getElementById('btn-pause');
const btnStop = document.getElementById('btn-stop');
const recordingTimer = document.getElementById('recording-timer');
const recordingStatus = document.getElementById('recording-status');

// ===== 전역 변수 =====
let selectedFiles = []; // 드래그/선택된 파일을 저장해두는 배열
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let silentGainNode = null;
let recordingBuffers = [];
let recordingLength = 0;
let recordingSampleRate = 44100;
let isRecording = false;
let recordingStartTime = 0;
let timerInterval = null;
let isPaused = false;
let pauseStartTime = 0;
let pausedDuration = 0;

// 클릭으로 파일 선택
if (dropzone) {
    dropzone.addEventListener('click', () => {
        if (hiddenInput) hiddenInput.click();
    });
}

// 드래그 기본 동작 막기
if (dropzone) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });
}

// 강조 스타일
if (dropzone) {
    ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, () => dropzone.classList.add('highlight'));
    });
    ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, () => dropzone.classList.remove('highlight'));
    });
}

// 드롭, input 선택 처리
['drop', 'change'].forEach(eventType => {
    const target = eventType === 'drop' ? dropzone : hiddenInput;
    if (!target) return;
    target.addEventListener(eventType, (e) => {
        const files = eventType === 'drop' ? e.dataTransfer.files : e.target.files;
        handleFiles(files);
    });
});

// 파일 처리
function handleFiles(fileList) {
    if (!fileList || fileList.length === 0) return;

    const file = fileList[0];      // 한 개만 사용
    const allowed = ["wav"];

    // 확장자 체크
    const ext = file.name.split(".").pop().toLowerCase();
    if (!allowed.includes(ext)) {
        alert("wav 파일만 업로드할 수 있습니다.");
        // 선택 초기화
        selectedFiles = [];
        preview.innerHTML = '';
        if (hiddenInput) hiddenInput.value = "";
        return;
    }

    // 업로드 확인
    const ok = confirm(`${file.name} (${Math.round(file.size / 1024)} KB)를 업로드하시겠습니까?`);
    if (!ok) {
        // 취소하면 선택/미리보기 초기화
        selectedFiles = [];
        preview.innerHTML = '';
        if (hiddenInput) hiddenInput.value = "";
        return;
    }
    selectedFiles = [file];

    // 미리보기 렌더
    if (preview) {
        preview.innerHTML = '';
        const item = document.createElement('div');
        item.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
        preview.appendChild(item);
    }
}

// ===== 녹음 기능 =====

// Attach recording handlers only if recording UI exists to avoid JS errors
if (btnRecord) {
    function resetRecordingBuffers() {
        recordingBuffers = [];
        recordingLength = 0;
    }

    function mergeBuffers(buffers, length) {
        const result = new Float32Array(length);
        let offset = 0;
        buffers.forEach((buffer) => {
            result.set(buffer, offset);
            offset += buffer.length;
        });
        return result;
    }

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    function floatTo16BitPCM(view, offset, input) {
        for (let i = 0; i < input.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, input[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }
    }

    function buildWavBlob(buffers, length, sampleRate) {
        const data = mergeBuffers(buffers, length);
        const buffer = new ArrayBuffer(44 + data.length * 2);
        const view = new DataView(buffer);

        writeString(view, 0, "RIFF");
        view.setUint32(4, 36 + data.length * 2, true);
        writeString(view, 8, "WAVE");
        writeString(view, 12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, "data");
        view.setUint32(40, data.length * 2, true);
        floatTo16BitPCM(view, 44, data);

        return new Blob([view], { type: "audio/wav" });
    }

    function cleanupRecorder() {
        if (processorNode) {
            processorNode.disconnect();
            processorNode.onaudioprocess = null;
        }
        if (sourceNode) {
            sourceNode.disconnect();
        }
        if (silentGainNode) {
            silentGainNode.disconnect();
        }
        if (audioContext) {
            audioContext.close();
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach((track) => track.stop());
        }
        processorNode = null;
        sourceNode = null;
        silentGainNode = null;
        audioContext = null;
        mediaStream = null;
    }

    function finalizeRecording() {
        if (recordingLength === 0) {
            alert("녹음된 내용이 없습니다.");
            return;
        }
        const audioBlob = buildWavBlob(
            recordingBuffers,
            recordingLength,
            recordingSampleRate
        );
        const audioFile = new File([audioBlob], `recording_${Date.now()}.wav`, {
            type: "audio/wav",
        });

        selectedFiles = [audioFile];
        if (preview) preview.innerHTML = "";
        const item = document.createElement("div");
        item.textContent = `${audioFile.name} (${Math.round(
            audioFile.size / 1024
        )} KB) - 녹음 완료`;
        if (preview) preview.appendChild(item);

        if (recordingStatus) {
            recordingStatus.textContent = "녹음이 완료되었습니다.";
            recordingStatus.classList.remove("active");
        }
    }
    // 타이머 업데이트
    function updateTimer() {
        const elapsed = Date.now() - recordingStartTime - (pausedDuration || 0);
        const minutes = Math.floor(elapsed / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        if (recordingTimer) recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }

    // 녹음 시작
    btnRecord.addEventListener('click', async () => {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            recordingSampleRate = audioContext.sampleRate;
            sourceNode = audioContext.createMediaStreamSource(mediaStream);
            processorNode = audioContext.createScriptProcessor(4096, 1, 1);
            silentGainNode = audioContext.createGain();
            silentGainNode.gain.value = 0;

            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }

            resetRecordingBuffers();
            isRecording = true;

            processorNode.onaudioprocess = (event) => {
                if (!isRecording || isPaused) return;
                const inputData = event.inputBuffer.getChannelData(0);
                recordingBuffers.push(new Float32Array(inputData));
                recordingLength += inputData.length;
            };

            sourceNode.connect(processorNode);
            processorNode.connect(silentGainNode);
            silentGainNode.connect(audioContext.destination);
            recordingStartTime = Date.now();
            // 초기화
            pausedDuration = 0;
            pauseStartTime = 0;
            isPaused = false;

            // 타이머 시작
            if (timerInterval) clearInterval(timerInterval);
            timerInterval = setInterval(updateTimer, 100);

            // UI 업데이트
            btnRecord.disabled = true;
            btnRecord.classList.add('recording');
            if (btnPause) btnPause.disabled = false;
            if (btnStop) btnStop.disabled = false;
            if (recordingStatus) {
                recordingStatus.textContent = '녹음 중...';
                recordingStatus.classList.add('active');
            }

        } catch (err) {
            console.error('녹음 시작 오류:', err);
            alert('마이크 접근 권한이 필요합니다.');
        }
    });

    // 일시정지/재개
    if (btnPause) {
        btnPause.addEventListener('click', () => {
            if (isRecording && !isPaused) {
                // pause: stop timer and record pause start
                if (audioContext && audioContext.state === 'running') {
                    audioContext.suspend();
                }
                clearInterval(timerInterval);
                pauseStartTime = Date.now();
                if (recordingStatus) recordingStatus.textContent = '일시정지됨';
                // UI: show paused state
                if (btnRecord) {
                    btnRecord.classList.remove('recording');
                    btnRecord.classList.add('paused');
                }
                isPaused = true;
            } else if (isRecording && isPaused) {
                // resume: accumulate paused duration and restart timer
                if (audioContext && audioContext.state === 'suspended') {
                    audioContext.resume();
                }
                if (pauseStartTime) {
                    pausedDuration += Date.now() - pauseStartTime;
                    pauseStartTime = 0;
                }
                if (timerInterval) clearInterval(timerInterval);
                timerInterval = setInterval(updateTimer, 100);
                if (recordingStatus) recordingStatus.textContent = '녹음 중...';
                // UI: restore recording state
                if (btnRecord) {
                    btnRecord.classList.add('recording');
                    btnRecord.classList.remove('paused');
                }
                isPaused = false;
            }
        });
    }

    // 녹음 중지
    if (btnStop) {
        btnStop.addEventListener('click', () => {
            if (isRecording) {
                isRecording = false;
                clearInterval(timerInterval);
                if (audioContext && audioContext.state === 'suspended') {
                    audioContext.resume();
                }
                finalizeRecording();
                cleanupRecorder();
                isPaused = false;

                // UI 초기화
                btnRecord.disabled = false;
                btnRecord.classList.remove('recording');
                btnRecord.classList.remove('paused');
                if (btnPause) btnPause.disabled = true;
                if (btnStop) btnStop.disabled = true;
            }
        });
    }

    // 페이지 이탈 경고: 녹음 중이거나 업로드되지 않은 파일이 있으면 경고
    window.addEventListener('beforeunload', function (e) {
        const recordingActive = isRecording || isPaused;
        const hasUnsaved = selectedFiles && selectedFiles.length > 0;
        if (recordingActive || hasUnsaved) {
            e.preventDefault();
            e.returnValue = '녹음 중이거나 업로드되지 않은 파일이 있습니다. 페이지를 떠나시겠습니까?';
            return e.returnValue;
        }
    });
}

// 업로드 버튼 클릭 시 실행
uploadBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    uploadBtn.disabled = true;
    if (!selectedFiles || selectedFiles.length === 0) {
        // 파일이 선택되지 않았으면 파일 선택 창을 연다
        if (hiddenInput) {
            hiddenInput.click();
            return;
        }
        alert('업로드할 파일이 없습니다');
        return;
    }
    const pathParts = window.location.pathname.split('/');
    const meetingId = pathParts[2];

    const formData = new FormData();

    formData.append('file', selectedFiles[0]);
    try {
        const res = await fetch(`/meetings/${meetingId}/upload/`, {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken,
            },
            body: formData,
        });

        const data = await res.json().catch(() => ({}));

        if (res.ok && data.ok !== false) {
            alert('업로드 성공!');
            // 필요하면 selectedFiles/preview 초기화
            selectedFiles = [];
            if (preview) preview.innerHTML = '';
            if (hiddenInput) hiddenInput.value = "";

            // 녹음 관련 초기화
            if (recordingTimer) recordingTimer.textContent = '00:00';
            if (recordingStatus) recordingStatus.textContent = '';

            window.location.href = `/meetings/${meetingId}/rendering/stt/`;
        } else {
            alert(`업로드 실패: ${data.error ?? res.statusText ?? '알 수 없는 오류'}`);
        }
    } catch (err) {
        console.error(err);
        alert(`업로드 중 오류가 발생했습니다: ${err.message || err}`);
    } finally {
    // 완료 후 다시 활성화
    uploadBtn.disabled = false;
  }
});
