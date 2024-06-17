# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

block_cipher = None

added_files = [
                ('C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\.venv\\Lib\\site-packages\\mindrove\\lib\\BoardController.dll', '.'),
                ('C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\.venv\\Lib\\site-packages\\mindrove\\lib\\DataHandler.dll', '.'),
                ('C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\.venv\\Lib\\site-packages\\mindrove\\lib\\MindRoveBluetooth.dll', '.'),
                ('C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\.venv\\Lib\\site-packages\\mindrove\\lib\\MLModule.dll', '.'),
                ('C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\.venv\\Lib\\site-packages\\mindrove\\lib\\simpleble-c.dll', '.')
              ]

binaries = []
hiddenimports = []
tmp_ret = collect_all('ml_dtypes')
added_files += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mindrove')
added_files += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=added_files,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

a.datas += [
                ("IDSystem Documentation v0.5.pdf", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\docs\\IDSystem Documentation v0.5.pdf", "DATA"),
                ("TitilliumWeb-Black.ttf", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\fonts\\TitilliumWeb-Black.ttf", "DATA"),
                ("TitilliumWeb-Bold.ttf", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\fonts\\TitilliumWeb-Bold.ttf", "DATA"),
                ("TitilliumWeb-BoldItalic.ttf", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\fonts\\TitilliumWeb-BoldItalic.ttf", "DATA"),
                ("TitilliumWeb-ExtraLight.ttf", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\fonts\\TitilliumWeb-ExtraLight.ttf", "DATA"),
                ("TitilliumWeb-ExtraLightItalic.ttf", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\fonts\\TitilliumWeb-ExtraLightItalic.ttf", "DATA"),
                ("TitilliumWeb-Italic.ttf", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\fonts\\TitilliumWeb-Italic.ttf", "DATA"),
                ("TitilliumWeb-Light.ttf", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\fonts\\TitilliumWeb-LightItalic.ttf", "DATA"),
                ("TitilliumWeb-LightItalic.ttf", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\fonts\\TitilliumWeb-Regular.ttf", "DATA"),
                ("TitilliumWeb-Regular.ttf", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\fonts\\TitilliumWeb-Regular.ttf", "DATA"),
                ("TitilliumWeb-SemiBold.ttf", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\fonts\\TitilliumWeb-SemiBold.ttf", "DATA"),
                ("TitilliumWeb-SemiBoldItalic.ttf", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\fonts\\TitilliumWeb-SemiBoldItalic.ttf", "DATA"),
                ("data-transfer.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\icons\\data-transfer.png", "DATA"),
                ("ethernet.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\icons\\ethernet.png", "DATA"),
                ("time.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\icons\\time.png", "DATA"),
                ("folder-open.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\icons\\folder-open.png", "DATA"),
                ("table.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\icons\\table.png", "DATA"),
                ("subject.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\icons\\subject.png", "DATA"),
                ("open.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\imgs\\open.png", "DATA"),
                ("close.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\imgs\\close.png", "DATA"),
                ("tripod_open.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\imgs\\tripod_open.png", "DATA"),
                ("tripod.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\imgs\\tripod.png", "DATA"),
                ("bottom_open.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\imgs\\bottom_open.png", "DATA"),
                ("bottom_close.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\imgs\\bottom_close.png", "DATA"),
                ("rest_label.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\imgs\\rest_label.png", "DATA"),
                ("demo.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\imgs\\demo.png", "DATA"),
                ("tasks.png", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\imgs\\tasks.png", "DATA"),
                ("demoMsg.txt", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\text\\demoMsg.txt", "DATA"),
                ("gettingStarted.txt", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\text\\gettingStarted.txt", "DATA"),
                ("mainInstructions.txt", "C:\\Users\\penafiel\\Documents\\intention_detection\\IntentionDetectionSystem\\V0.5.2\\dev\\assets\\text\\mainInstructions.txt", "DATA")
               ]


pyz = PYZ(a.pure, a.zipped_data,
         cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    exclude_binaries=False,
    name='IDSystem V0.5.2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='IDSystem V0.5.2',
)
