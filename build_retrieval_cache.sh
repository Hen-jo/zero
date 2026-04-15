#!/bin/bash
# retrieval server 첫 실행용 캐시 빌더
# HuggingFace datasets 캐시 + FAISS 인덱스 로딩 확인
# 완료 후 Ctrl+C로 종료하면 됨

set -e

export HF_DATASETS_CACHE=/input/cache/huggingface

echo "[INFO] retrieval server를 포그라운드로 실행합니다."
echo "[INFO] 처음 실행 시 corpus 캐시 생성에 30분~1시간 소요될 수 있습니다."
echo "[INFO] 'Application startup complete' 메시지 확인 후 Ctrl+C로 종료하세요."
echo ""

PYTHONUNBUFFERED=1 python search/retrieval_server.py \
    --index_path='/input/data/Drzero/corpus/e5_Flat.index' \
    --corpus_path='/input/data/Drzero/corpus/wiki-18.jsonl' \
    --retriever_model='/input/model/e5-base-v2' \
    --retriever_name='e5' \
    --faiss_gpu \
    --topk 3
