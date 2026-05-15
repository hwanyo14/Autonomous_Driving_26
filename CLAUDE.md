# Agent Instructions For Claude Code

- 항상 필요한만큼만 최소 수정으로 진행할 것
- 코드 구현의 redundant를 최대한 줄이고 가독성 좋게 작성할 것
- 구현이나 로직이 애매한 부분은 사용자에게 역으로 질문하여 요청할 것
- 불필요한 파일 탐색이나 로드는 최대한 지양할 것 (반드시 필요한 경우에만 허용)
- 불필요한 파일 탐색을 예방하기 위해 `PROJECT_STRUCTURE.md`를 통해 repo 구조를 파악할 것
- 새로운 파일을 생성하여 작성하거나 프로젝트 tree 구조가 변경되는 경우 `PROJECT_STRUCTURE.md` 파일에 내용을 업데이트할 것
- 코드 수정 및 구현 내용을 매번 `CHANGELOG.md`에 기록할 것, 날짜와 시간, 핵심 내용, 주요 변경사항 등등 (초기에 change_log 파일이 없다면 생성할 것)
- 이전 기록 및 구현 내용이 필요한 경우 `CHANGELOG.md`를 참고할 것
- 코드 구현 중 놓친 부분이 있거나 실패한 지점, 혹은 추후 구현에서 반드시 알아야 할법한 주요 정보나 경고, notation은 `NOTES.md`에 정리할 것 (파일이 없다면 생성할 것)
- 코드 구현 중 새로운 인자나 설정값 추가 및 변경 시 `projects/configs/baselines/EfficientOCF_V1.1_1gpu.py`와 `projects/occ_plugin/occupancy/detectors/efficientocf_config.py`를 모두 고려할 것