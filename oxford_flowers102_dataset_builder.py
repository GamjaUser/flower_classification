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

# Oxford 102 Flowers Dataset을 로드 및 처리
# 해당 빌더는 데이터 다운로드, 전처리, tensorflow와 호환되는 형식으로 데이터셋 제공

"""Oxford 102 Category Flower Dataset."""

import os

from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf  #lazy_imports_utils : tensorflow 및 기타 라이브러리 동적으로 가져오는 유틸리티 import
import tensorflow_datasets.public_api as tfds


# import tensorflow_datasets as tfds
# print(tfds.list_builders())  # 사용 가능한 데이터셋 목록 출력

_BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"

#102개 꽃 이름 목록, 라벨 번호(0~101)와 매핑 0->pink primrose..
_NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]


class Builder(tfds.core.GeneratorBasedBuilder): 
  """Oxford 102 category flower dataset."""

  VERSION = tfds.core.Version("2.1.1") #dataset 버전

  def _info(self):
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(), #이미지 데이터
            "label": tfds.features.ClassLabel(names=_NAMES), #클래스라벨
            "file_name": tfds.features.Text(), #이미지 파일명
        }),
        supervised_keys=("image", "label"), #지도 학습 키, 모델 학습 시 입력(image)과 출력(label) 키를 정의
        homepage=_BASE_URL, #데이터셋 홈페이지 url
    )

  #데이터를 다운로드하고 분할 정보 로드 
  def _split_generators(self, dl_manager): 
    """Returns SplitGenerators."""
    # Download images and annotations that come in separate archives.
    # Note, that the extension of archives is .tar.gz even though the actual
    # archives format is uncompressed tar.
    dl_paths = dl_manager.download_and_extract({ #원본 데이터셋 파일을 다운로드 및 압축 풀기, 추출된 파일 경로 dl_paths에 저장
        "images": os.path.join(_BASE_URL, "102flowers.tgz"), 
        "labels": os.path.join(_BASE_URL, "imagelabels.mat"), 
        "setid": os.path.join(_BASE_URL, "setid.mat"), 
    })

    gen_kwargs = dict( # _generate_examples에 필요한 인자를 저장하는 딕셔너리 / 훈련, 테스트, 검증 세트로 분리할 때 사용
        images_dir_path=os.path.join(dl_paths["images"], "jpg"), #이미지 디렉토리 경로
        labels_path=dl_paths["labels"], #각 이미지의 라벨 경로
        setid_path=dl_paths["setid"], # 데이터 분할 정보 파일 경로
    )

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN, #훈련
            gen_kwargs=dict(split_name="trnid", **gen_kwargs),
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST, #테스트
            gen_kwargs=dict(split_name="tstid", **gen_kwargs),
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION, #검증
            gen_kwargs=dict(split_name="valid", **gen_kwargs),
        ),
    ]

  #이미지, 라벨, 파일 이름 정보를 구성
  def _generate_examples(
      self, images_dir_path, labels_path, setid_path, split_name
  ):
    """Yields examples."""
    with tf.io.gfile.GFile(labels_path, "rb") as f: #TensorFlow의 파일 읽기 유틸리티. 
      labels = tfds.core.lazy_imports.scipy.io.loadmat(f)["labels"][0] # scipy.io.loadmat(f) : .mat파일을 파이썬으로 읽기 위한 함수 / 라벨 배열 0부터 변환
    with tf.io.gfile.GFile(setid_path, "rb") as f: 
      examples = tfds.core.lazy_imports.scipy.io.loadmat(f)[split_name][0] #split_name에 따라 훈련, 테스트, 검증 선택

    for image_id in examples:
      file_name = "image_%05d.jpg" % image_id #이미지 파일 이름 생성
      record = { #한개의 데이터 항목을 나타내는 딕셔너리
          "image": os.path.join(images_dir_path, file_name),
          "label": labels[image_id - 1] - 1, #0부터 시작하도록 -1로 변환
          "file_name": file_name,
      }
      yield file_name, record #데이터 생성기로 반환 / 키 : 이미지 파일 이름, 값 : 이미지 라벨 파일 이름 정보를 포함한 딕셔너리