from setuptools import setup, find_packages


setup(
    name='bert_reranker',
    version='0.2.1',
    packages=find_packages(include=['bert_reranker', 'bert_reranker.*']),
    license='MIT',
    author='Mirko Bronzi',
    author_email='m.bronzi@gmail.com',
    url='https://github.com/mirkobronzi/bert_reranker',
    python_requires='>=3.7',
    install_requires=[
        'flake8', 'tqdm', 'pyyaml>=5.3', 'pytest', 'numpy>=1.16.4', 'pytest', 'pandas', 'nltk',
        'torch==1.4.0', 'transformers==3.0.2', 'pytorch-lightning==0.8.5', 'scikit-learn',
        'pandas', 'wandb', 'setuptools>=41.0.0',
        'orion'],
    entry_points={
        'console_scripts': [
            'main=bert_reranker.main:main',
            'generation=bert_reranker.generation:main',
            'multitask=bert_reranker.multitask:main',
            'main_tune=bert_reranker.main_tune:main'
            'rating_multitask=bert_reranker.rating_multitask:main',
            'justification_multitask=bert_reranker.multitask_justification_encoding:main',
            'train_outlier_detector=bert_reranker.models.sklearn_outliers_model:main',
            'fix_annotations=bert_reranker.scripts.fix_annotations:main'
        ],
    }
)