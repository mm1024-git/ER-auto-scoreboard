# PyInstaller 설정. 윈도우에서 아래 한 줄로 만든다.
#
#     pyinstaller er_scoreboard.spec
#
# dist\ER_score\ER_score.exe 가 나온다. 폴더째 옮겨서 쓰면 되고, 같은 폴더에
# config.json과 digits.npz(또는 templates 폴더)를 두면 그대로 읽는다.
#
# 한 파일(.exe 하나)로 묶지 않는 이유가 있다. 한 파일로 묶으면 실행할 때마다 임시
# 폴더에 풀어서 시작이 느리고, 설정과 학습한 가중치를 옆에 두고 고치기도 번거롭다.

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("config.json", "."),   # 전체 화면 기준으로 잡아 둔 영역
    ],
    hiddenimports=[
        # 창에서 고르는 것들이라 정적 분석으로는 안 잡히는 것들
        "tkinterweb",
        "windows_capture",
        "mss",
        "win32gui",
        "win32process",
        "PIL._tkinter_finder",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "pandas", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ER_score",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # 창만 뜬다. 오류를 콘솔로 보려면 True로 바꾼다
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="ER_score",
)
