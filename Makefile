.PHONY: up down seed agent logs ps install

# Start the full stack (postgres, cube, library, MCP servers)
up:
	docker compose up -d --build
	@echo ""
	@echo "Waiting for services to be ready..."
	@until curl -sf http://localhost:3001/health > /dev/null 2>&1; do sleep 2; done
	@echo "  library     OK  (http://localhost:3001)"
	@until curl -sf http://localhost:4000/cubejs-api/v1/meta > /dev/null 2>&1; do sleep 2; done
	@echo "  cube        OK  (http://localhost:4000)"
	@until docker compose ps cube-mcp | grep -q "healthy"; do sleep 2; done
	@echo "  cube-mcp    OK  (http://localhost:5001/sse)"
	@until docker compose ps library-mcp | grep -q "healthy"; do sleep 2; done
	@echo "  library-mcp OK  (http://localhost:5002/sse)"
	@echo ""
	@echo "Seeding library with cube configs..."
	@$(MAKE) seed
	@echo ""
	@echo "Stack ready. Run:  make agent"

# Stop all services and remove volumes
down:
	docker compose down -v

# Seed the library with e-commerce cube configs
seed:
	cd library && python seed.py

# Run the interactive reporting agent (requires agent/.env with ANTHROPIC_API_KEY)
agent:
	@if [ ! -f agent/.env ]; then \
	  cp .env.example agent/.env; \
	  echo "[!] Created agent/.env — add your ANTHROPIC_API_KEY then re-run make agent"; \
	  exit 1; \
	fi
	cd agent && python main.py

# Create virtualenv and install agent dependencies
install:
	python3 -m venv .venv
	.venv/bin/pip install -r agent/requirements.txt

# Run the web UI (activates venv automatically)
ui:
	ANTHROPIC_API_KEY=$$(grep ANTHROPIC_API_KEY agent/.env | cut -d= -f2) \
	  .venv/bin/python agent/ui.py

# Tail logs from all services
logs:
	docker compose logs -f

# Show service status and ports
ps:
	@echo ""
	@echo "Service       Port   Status"
	@echo "──────────────────────────"
	@docker compose ps --format "table {{.Service}}\t{{.Ports}}\t{{.Status}}" 2>/dev/null || docker compose ps
	@echo ""
	@echo "Cube Playground: http://localhost:4000"
	@echo "Library API:     http://localhost:3001"
	@echo "Cube MCP:        http://localhost:5001/sse"
	@echo "Library MCP:     http://localhost:5002/sse"
	@echo "Chart viewer:    http://localhost:8080"
