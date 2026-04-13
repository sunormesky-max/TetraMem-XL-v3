from tetrahedron_memory.router import create_app

app = create_app(dimension=3, precision='fast')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
