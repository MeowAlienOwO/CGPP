from bpp1d.hello import hello

def test_hello():
    name = "your name"
    assert hello(name) == f"hello {name}" 
