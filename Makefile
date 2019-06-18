build_release:
	python setup.py sdist

publish_release:
	twine upload dist/gym_dummy-${VERSION}.tar.gz
