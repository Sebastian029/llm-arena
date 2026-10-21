# Backend rules
- Endpoints in api/ are thin: validation and a call to a function from services/.
- Every model provider implements the same interface in adapters/. The rest of the code does not know which provider it uses.
- Celery tasks must be idempotent (running again does not corrupt the result) and save progress so they can be resumed.
- Tests: always replace providers with mocks. Use pytest fixtures for the test database.
- Cost calculation and budget limit checks live in one place in services/, do not scatter this logic.
