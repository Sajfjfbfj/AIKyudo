// ===== フォント切り替え =====
        document.querySelectorAll('input[name="certFont"]').forEach(radio => {
            radio.addEventListener('change', () => {
                const font = radio.value;
                certContainer.style.fontFamily = `'${font}', serif`;
                document.querySelectorAll('.font-option').forEach(el => el.classList.remove('selected'));
                radio.closest('.font-option').classList.add('selected');
            });
        });

        // ===== パネル開閉 =====
        let panelOpen = false;

        function togglePanel() {
            panelOpen = !panelOpen;
            const panel = document.getElementById('controlsPanel');
            const label = document.getElementById('panelBtnLabel');
            if (panelOpen) {
                panel.classList.add('panel-open');
                label.textContent = '閉じる';
            } else {
                panel.classList.remove('panel-open');
                label.textContent = '入力パネルを開く';
                setTimeout(scaleCertificate, 300);
            }
        }

        document.getElementById('previewArea').addEventListener('click', function () {
            if (panelOpen && window.innerWidth <= 768) {
                panelOpen = false;
                document.getElementById('controlsPanel').classList.remove('panel-open');
                document.getElementById('panelBtnLabel').textContent = '入力パネルを開く';
                setTimeout(scaleCertificate, 300);
            }
        });

        // ===== 賞状スケーリング =====
        const CERT_WIDTH_PX  = Math.round(420 * 3.7795);
        const CERT_HEIGHT_PX = Math.round(297 * 3.7795);

        function scaleCertificate() {
            if (window.innerWidth <= 768 && panelOpen) return;
            const outer   = document.getElementById('previewOuter');
            const wrapper = document.getElementById('previewWrapper');
            const area    = document.getElementById('previewArea');
            const style   = getComputedStyle(area);
            const padH    = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
            const padV    = parseFloat(style.paddingTop)  + parseFloat(style.paddingBottom);
            const areaW   = area.clientWidth  - padH;
            const areaH   = area.clientHeight - padV;
            const scale   = Math.min(areaW / CERT_WIDTH_PX, areaH / CERT_HEIGHT_PX, 1);
            wrapper.style.width          = CERT_WIDTH_PX  + 'px';
            wrapper.style.height         = CERT_HEIGHT_PX + 'px';
            wrapper.style.transform      = `scale(${scale})`;
            wrapper.style.transformOrigin = 'top left';
            outer.style.width  = Math.round(CERT_WIDTH_PX  * scale) + 'px';
            outer.style.height = Math.round(CERT_HEIGHT_PX * scale) + 'px';
        }

        window.addEventListener('resize', scaleCertificate);
        window.addEventListener('DOMContentLoaded', scaleCertificate);
        document.fonts.ready.then(scaleCertificate);

        // ===== 入力値と表示を同期 =====
        const inputs = {
            department:    document.getElementById('inputDepartment'),
            rank:          document.getElementById('inputRank'),
            teamName:      document.getElementById('inputTeamName'),
            recipient1:    document.getElementById('inputRecipient1'),
            recipient2:    document.getElementById('inputRecipient2'),
            recipient3:    document.getElementById('inputRecipient3'),
            organizerName: document.getElementById('inputOrganizerName'),
            tournamentName:document.getElementById('inputTournamentName'),
            nakanoNum:     document.getElementById('inputNakanoNum'),
            date:          document.getElementById('inputDate'),
            organization:  document.getElementById('inputOrganization'),
            sender:        document.getElementById('inputSender')
        };

        const displays = {
            department:   document.getElementById('dispDepartment'),
            rank:         document.getElementById('dispRank'),
            teamName:     document.getElementById('dispTeamName'),
            recipient1:   document.getElementById('dispRecipient1'),
            recipient2:   document.getElementById('dispRecipient2'),
            recipient3:   document.getElementById('dispRecipient3'),
            body:         document.getElementById('dispBody'),
            date:         document.getElementById('dispDate'),
            organization: document.getElementById('dispOrganization'),
            sender:       document.getElementById('dispSender')
        };

        const certContainer = document.querySelector('.certificate');

        // 現在のサブモードを取得
        function getSubMode() {
            const el = document.querySelector('input[name="subMode"]:checked');
            return el ? el.value : 'normal';
        }

        // 本文更新
        function updateBody() {
            if (getSubMode() === 'nakano') {
                // ── 中野杯固定本文 ──
                const num  = inputs.nakanoNum.value || '○○';
                displays.body.innerText =
                    `右は茨城県弓道連盟主催\n第${num}回中野慶吉杯記念弓道大会において頭書の成績をおさめたのでここに表彰します`;
                certContainer.classList.add('long-tournament');
            } else {
                // ── 通常モード ──
                const organizer     = inputs.organizerName.value  || '○○';
                const tournament    = inputs.tournamentName.value || '○○大会';
                const organizerType = document.querySelector('input[name="organizerType"]:checked').value;
                const isLong        = tournament.length > 10 || organizer.length > 8;

                displays.body.innerText = tournament.length > 10
                    ? `右は${organizer}${organizerType}\n${tournament}において頭書の成績を収めたのでここにこれを賞します`
                    : `右は${organizer}${organizerType}\n${tournament}で頭書の成績を収めたのでここにこれを賞します`;

                if (isLong) {
                    certContainer.classList.add('long-tournament');
                } else {
                    certContainer.classList.remove('long-tournament');
                }
            }
        }

        // 個人戦サブモード切替（通常 ↔ 中野杯）
        function toggleSubMode() {
            const isNakano = getSubMode() === 'nakano';
            document.getElementById('normalBodyInputs').style.display = isNakano ? 'none'  : 'block';
            document.getElementById('nakanoBodyInputs').style.display = isNakano ? 'block' : 'none';
            updateBody();
        }

        // 個人戦 / 団体戦 切替
        function toggleMode() {
            const isTeam = document.querySelector('input[name="mode"][value="team"]').checked;
            const teamInputs = document.querySelectorAll('.team-only');
            teamInputs.forEach(el => el.style.display = isTeam ? 'block' : 'none');

            // サブモードは個人戦のみ表示
            document.getElementById('singleSubMode').style.display = isTeam ? 'none' : 'block';

            if (isTeam) {
                // 団体戦では通常入力を表示、中野杯入力を隠す
                document.getElementById('normalBodyInputs').style.display = 'block';
                document.getElementById('nakanoBodyInputs').style.display = 'none';
                // subModeを通常にリセット
                document.querySelector('input[name="subMode"][value="normal"]').checked = true;

                inputs.department.parentElement.style.display = 'none';
                certContainer.classList.remove('has-department');
                certContainer.classList.add('team-mode');
            } else {
                // 個人戦に戻ったらsubModeの状態を再適用
                toggleSubMode();

                inputs.department.parentElement.style.display = 'block';
                if (inputs.department.value && inputs.department.value.trim() !== '') {
                    certContainer.classList.add('has-department');
                }
                certContainer.classList.remove('team-mode');
            }

            if (!isTeam) {
                document.getElementById('divRecipient2').style.display = 'none';
                document.getElementById('divRecipient3').style.display = 'none';
                displays.teamName.style.display = 'none';
            } else {
                if (inputs.recipient2.value) document.getElementById('divRecipient2').style.display = 'block';
                if (inputs.recipient3.value) document.getElementById('divRecipient3').style.display = 'block';
                if (inputs.teamName.value)   displays.teamName.style.display = 'block';
            }

            checkMultiRecipients();
            updateBody();
        }

        // イベントリスナー一括登録
        Object.keys(inputs).forEach(key => {
            inputs[key].addEventListener('input', (e) => {
                if (['organizerName', 'tournamentName', 'nakanoNum'].includes(key)) {
                    updateBody();
                } else {
                    if (displays[key]) displays[key].innerText = e.target.value;
                }

                if (key === 'recipient2') {
                    document.getElementById('divRecipient2').style.display = e.target.value ? 'block' : 'none';
                    checkMultiRecipients();
                }
                if (key === 'recipient3') {
                    document.getElementById('divRecipient3').style.display = e.target.value ? 'block' : 'none';
                    checkMultiRecipients();
                }
                if (key === 'teamName') {
                    displays.teamName.style.display = e.target.value ? 'block' : 'none';
                }
                if (key === 'department') {
                    if (e.target.value && e.target.value.trim() !== '') {
                        certContainer.classList.add('has-department');
                    } else {
                        certContainer.classList.remove('has-department');
                    }
                }
            });
        });

        // 複数名チェック
        function checkMultiRecipients() {
            const isTeam   = document.querySelector('input[name="mode"][value="team"]').checked;
            const hasMulti = inputs.recipient2.value.trim() !== '' || inputs.recipient3.value.trim() !== '';
            if (isTeam && hasMulti) {
                certContainer.classList.add('multi-recipients');
            } else {
                certContainer.classList.remove('multi-recipients');
            }
            if (isTeam && inputs.recipient3.value.trim() !== '') {
                certContainer.classList.add('three-recipients');
            } else {
                certContainer.classList.remove('three-recipients');
            }
        }

        // PDF保存
        function savePDF() {
            if (!confirm("【PDF保存の手順】\n\n1. この後開く印刷画面で、送信先を「PDFに保存」に変更してください。\n2. 「保存」ボタンを押すとPDFファイルがダウンロードされます。\n\n※この方法なら画質が劣化せず、文字も綺麗に保存されます。\n\n進みますか？")) return;
            window.print();
        }
