from setuptools import find_packages, setup

setup(
    name="vipact_multimodal_brain",
    version="0.1.0",
    package_dir={"": ".."},
    packages=find_packages(where="..", include=["multimodal_brain", "multimodal_brain.*"]),
    install_requires=[
        "numpy>=1.22",
        "Pillow>=9.0",
    ],
    extras_require={
        "dino": ["transformers>=4.40"],
    },
    entry_points={
        "console_scripts": [
            "dino-detect-image=multimodal_brain.dino_detector.detect_image:main",
            "dino-visualize-detections=multimodal_brain.dino_detector.visualize_detections:main",
            "llm-parse-and-select=multimodal_brain.llm_parser.parse_and_select:main",
            "mask-from-llm-box=multimodal_brain.mask_generator.mask_from_llm_box:main",
        ]
    },
)
