# coding=utf-8
# Copyright 2024 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TODO(oxford_102_flowers): Add a description here."""

from tensorflow_datasets import testing #데이터셋 검증하는 데 사용
from tensorflow_datasets.datasets.oxford_flowers102 import oxford_flowers102_dataset_builder #builder 클래스 import

# testing.DatasetBuilderTestCase 상속
class OxfordFlowers102Test(testing.DatasetBuilderTestCase):
  DATASET_CLASS = oxford_flowers102_dataset_builder.Builder
  SPLITS = { # 훈련, 데이터, 검증의 샘플 개수 정의
      "train": 1020,
      "test": 6149,
      "validation": 1020,
  }

  # 다운로드 결과 파일 이름 정의
  DL_EXTRACT_RESULT = {
      "images": "images",
      "labels": "imagelabels.mat",
      "setid": "setid.mat",
  }


if __name__ == "__main__": # 테스트 실행 메인 함수
  testing.test_main()