# HAEINDEX

웹 화면에서 PDF를 올리고 자연어로 질문하는 문서 검색·답변 프로젝트입니다.
PDF를 업로드하면 자동으로 문서를 분석하고 임베딩을 생성해 OpenSearch에 색인합니다.
색인이 완료되면 바로 질문할 수 있으며, 답변의 출처를 눌러 원본 PDF 페이지를 확인할 수 있습니다.

OpenSearch의 키워드·벡터 검색과 AWS Bedrock을 사용합니다. 웹 서버는 Python,
화면은 HTML·CSS·JavaScript로 구성되어 별도 프론트엔드 빌드는 필요하지 않습니다.

## 실행 방법

Python 3.13 이상, [uv](https://docs.astral.sh/uv/getting-started/installation/), 실행 중인 Docker와
AWS Bedrock API 키가 필요합니다. 사용할 AWS 리전에서 생성 모델과 Titan 임베딩 모델에
대한 호출 권한을 준비합니다.

### 1. 설치

```bash
git clone https://github.com/haein55/haeindex.git
cd haeindex
uv sync --locked
```

### 2. 검색 서버 실행

한국어 분석용 `analysis-nori` 플러그인이 설치된 OpenSearch를 `localhost:9200`에서 사용합니다.
처음 실행할 때 Docker 이미지를 만들고 컨테이너를 시작합니다.

```bash
docker build -t haeindex-opensearch - <<'DOCKERFILE'
FROM opensearchproject/opensearch:3.2.0
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch analysis-nori
DOCKERFILE

docker run -d --name haeindex-opensearch \
  -p 127.0.0.1:9200:9200 \
  -e discovery.type=single-node \
  -e DISABLE_INSTALL_DEMO_CONFIG=true \
  -e DISABLE_SECURITY_PLUGIN=true \
  -e "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m" \
  -v haeindex-opensearch-data:/usr/share/opensearch/data \
  haeindex-opensearch
```

이후에는 `docker start haeindex-opensearch`로 다시 시작합니다.
`curl http://localhost:9200`에 JSON 응답이 오면 준비가 완료된 것입니다.

### 3. Bedrock 설정과 웹 실행

아래 명령은 키를 화면이나 셸 명령 기록에 남기지 않고 입력받습니다. 키는 실행할 터미널의
환경 변수로만 설정하며, 소스 코드에 넣지 않습니다.

```bash
export BEDROCK_REGION=us-east-1
export AWS_BEARER_TOKEN_BEDROCK="$(uv run python -c 'import getpass; print(getpass.getpass("Bedrock API key: "))')"
uv run haeindex serve
```

기본 모델은 답변·분석에 `us.openai.gpt-5.6-sol`, 비전에 Claude Sonnet 4.5,
색인 보강에 Claude Haiku 4.5, 임베딩에 Titan Text Embeddings V2를 사용합니다.
계정에서 이용 가능한 모델로 바꾸려면 서버 실행 전에 `HAEINDEX_BEDROCK_ANSWER_MODEL`,
`HAEINDEX_BEDROCK_ANALYSIS_MODEL`, `HAEINDEX_BEDROCK_VISION_MODEL`,
`HAEINDEX_BEDROCK_ENRICH_MODEL`을 설정합니다. OpenAI 모델도 Bedrock을 통해 호출합니다.

### 4. 화면에서 PDF 추가하고 질문하기

1. 브라우저에서 **http://127.0.0.1:8787**을 엽니다.
2. **＋ PDF 추가**를 눌러 PDF를 선택합니다. 여러 파일도 한 번에 선택할 수 있습니다.
3. 문서 분석·임베딩·색인이 자동으로 진행됩니다. 문서 목록에 **색인 완료**가 표시될 때까지 기다립니다.
4. 문서를 선택하고 질문을 입력한 뒤 **보내기**를 누릅니다. 문서를 선택하지 않으면 전체 문서에서 관련 내용을 찾습니다.
5. 답변의 출처 번호를 눌러 원본 PDF 페이지를 확인합니다.

별도의 CLI 색인 명령이나 **선택 문서 검색 보강** 실행 없이 업로드한 문서를 검색할 수 있습니다.
검색 보강은 선택 사항입니다. 파일 저장 후 색인에 실패한 문서는 목록의 **색인** 버튼으로 다시 처리할 수 있습니다.

파일당 최대 32MB의 PDF를 지원하며, 여러 파일은 순서대로 처리합니다. 같은 이름의 문서는
덮어쓰지 않습니다. 텍스트가 없는 스캔 PDF는 먼저 OCR 처리해야 합니다.

업로드한 PDF는 `data/inbox/`, 처리 결과와 캐시는 `work/`에 저장되고 Git에 포함되지 않습니다.
웹 서버는 로컬에서 실행되며, 질문·문서 텍스트·필요한 PDF 페이지는 처리 과정에서 Bedrock으로 전송됩니다.
