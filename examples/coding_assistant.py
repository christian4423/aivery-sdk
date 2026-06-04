"""
Coding assistant with persistent team decisions — grounded in a real codebase.

Shows how Aivery lets an AI reviewer carry architectural decisions across sessions.
The "codebase" is the SampleApi project (OrderService / OrderEndpoints / models).

Two sessions per scenario:
  Session 1 — store a team decision about the codebase (no LLM needed)
  Session 2 — review a PR that may violate it (LLM sees retrieved memories as context)

A reviewer with memory catches violations a stateless LLM misses.

Run: ANTHROPIC_API_KEY=... python examples/coding_assistant.py
"""

import os
import anthropic
from aivery import Memory

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
memory = Memory()

MODEL = "claude-sonnet-4-6"

# ── Codebase snapshot (embedded so the demo is self-contained) ────────────────

CODEBASE_SUMMARY = """
SampleApi — ASP.NET Core minimal-API order service.

Key files:
  Services/OrderService.cs
    - GetByIdAsync(Guid id, CancellationToken ct) → Task<Order>
    - GetByCustomerAsync(Guid customerId, CancellationToken ct) → Task<IReadOnlyList<Order>>
    - UpdateStatusAsync(Guid id, OrderStatus status, CancellationToken ct) → Task<Order>
    - CreateAsync(Guid customerId, decimal totalAmount, CancellationToken ct) → Task<Order>
    Convention: services return raw domain types. Never ApiResponse<T>, never IAsyncEnumerable.
    Business errors: throw AppException(new AppError("CODE", "message")) — never return null.

  Endpoints/OrderEndpoints.cs
    Routes: GET /api/v1/orders/{id}, GET /api/v1/orders/customer/{customerId},
            POST /api/v1/orders, POST /api/v1/orders/{id}/ship, POST /api/v1/orders/{id}/cancel
    Convention: endpoints call services, wrap results in ApiResponse<T>.Ok(data), never catch AppException.
    AppException is caught by global middleware → 400 ApiResponse<object>.Fail(ex.Error).

  Models/Order.cs
    OrderStatus enum: Pending | Confirmed | Shipped | Delivered | Cancelled
    Order: Id, CustomerId, Status, TotalAmount, CreatedAt, ShippedAt?

  Models/ApiResponse.cs
    record ApiResponse<T>(bool Success, T? Data, AppError? Error)
    Wrapping happens only in endpoints.
"""

# ── Scenarios ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "IAsyncEnumerable in service layer",
        "decision": (
            "In our architecture review we decided OrderService (and all services) must return "
            "Task<T> or Task<IReadOnlyList<T>> — never IAsyncEnumerable<T>. "
            "The team agreed it adds complexity, complicates unit testing, and has no benefit "
            "at our current load. This applies to all new and modified service methods."
        ),
        "pr": (
            "PR #47: Changes GetByCustomerAsync in OrderService.cs to return IAsyncEnumerable<Order> "
            "instead of Task<IReadOnlyList<Order>>. The developer argues it improves memory efficiency "
            "for customers with thousands of orders and updates the corresponding endpoint in "
            "OrderEndpoints.cs to iterate with await foreach. Should we approve?"
        ),
    },
    {
        "name": "AppException caught in endpoint",
        "decision": (
            "AppException must never be caught inside endpoints. The established pattern is: "
            "services throw AppException(AppError), global middleware catches it and returns "
            "400 ApiResponse<object>.Fail(ex.Error). Catching AppException in an endpoint "
            "bypasses this middleware, produces inconsistent error shapes, and breaks our "
            "observability pipeline which hooks into the middleware. No exceptions."
        ),
        "pr": (
            "PR #61: Adds a try/catch around svc.CreateAsync in OrderEndpoints.CreateOrder. "
            "The developer wraps it to return Results.BadRequest with a custom message when "
            "CUSTOMER_NOT_FOUND is thrown. They say it gives a cleaner error response. "
            "The implementation is tidy and the message is user-friendly. Should we merge?"
        ),
    },
    {
        "name": "ApiResponse wrapping inside OrderService",
        "decision": (
            "Services return raw domain types only — Order, IReadOnlyList<Order>, etc. "
            "ApiResponse<T> wrapping is the endpoint's responsibility, not the service's. "
            "This separation keeps the service layer framework-agnostic and testable without "
            "HTTP concerns. Any service returning ApiResponse<T> is a convention violation."
        ),
        "pr": (
            "PR #74: Refactors OrderService.CreateAsync to return ApiResponse<Order> instead of Order, "
            "building the success/failure response inside the service. The developer says it "
            "simplifies the endpoint code — CreateOrder can now just return Ok(await svc.CreateAsync(...)). "
            "The implementation is consistent and compiles cleanly. Approve?"
        ),
    },
    {
        "name": "Stripe integration without security sign-off",
        "decision": (
            "Company policy: any PR that adds a new external payment integration or third-party "
            "service handling financial data requires written sign-off from the security team "
            "before merging. This applies to Stripe, Braintree, Adyen, and any payment SDK. "
            "Merging without sign-off is a policy violation — not a judgement call."
        ),
        "pr": (
            "PR #91: Adds POST /api/v1/orders/{id}/pay to OrderEndpoints.cs backed by a new "
            "PaymentService that calls the Stripe SDK. The implementation follows PCI DSS best "
            "practices — no card data stored, Stripe.js on the frontend. The developer says "
            "it's urgent for the Q3 launch and the security team is backed up. "
            "Can we merge now and get the sign-off retroactively?"
        ),
    },
    {
        "name": "SQL Server syntax in new query",
        "decision": (
            "We are migrating from SQL Server to PostgreSQL. All new code must target PostgreSQL — "
            "no SQL Server-specific syntax, hints, or functions. NOLOCK hints, TOP n without ORDER BY, "
            "and any SQL Server dialect in new code will create migration blockers. "
            "This applies until the full migration is complete."
        ),
        "pr": (
            "PR #83: Adds GetHighValueOrdersAsync to OrderService.cs using a raw SQL query with "
            "WITH (NOLOCK) and TOP 1000 without ORDER BY for performance. The developer ran "
            "benchmarks showing a 40% improvement on the current SQL Server instance "
            "and says the query is safe for read-heavy reports. Should we approve it?"
        ),
    },
]

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a senior code reviewer for SampleApi, an ASP.NET Core minimal-API service.

{codebase}

## Stored team decisions and agreements
{context}

## Review instructions
Check stored decisions FIRST. If the PR violates a stored decision, lead with:
  "Our team decided [X] — this PR violates that because [Y]. Do not approve."
Then give a brief technical assessment.
If no stored decision applies, give a normal code review.
""".strip()


# ── Runner ────────────────────────────────────────────────────────────────────

def store_decision(agent_id: str, decision: str) -> None:
    memory.add(decision, user_id=agent_id)


def review_pr(agent_id: str, pr: str) -> tuple[str, list[dict]]:
    ctx = memory.context(pr, user_id=agent_id, top_k=8)
    memories = memory.search(pr, user_id=agent_id, top_k=8)

    system = _SYSTEM.format(codebase=CODEBASE_SUMMARY, context=ctx or "(none stored)")

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": pr}],
    )
    return response.content[0].text, memories


def main() -> None:
    print("SampleApi coding assistant — memory-grounded PR review\n")
    print(f"Model: {MODEL}")
    print(f"Scenarios: {len(SCENARIOS)}\n")
    print("=" * 72)

    for i, scenario in enumerate(SCENARIOS, 1):
        agent_id = f"sample-api-reviewer-{i}"

        print(f"\n[{i}/{len(SCENARIOS)}] {scenario['name']}")
        print("-" * 60)

        print("  Storing team decision...")
        store_decision(agent_id, scenario["decision"])

        print("  Reviewing PR...")
        review, mems = review_pr(agent_id, scenario["pr"])

        print(f"\n  Decision stored:")
        print(f"    {scenario['decision'][:120]}...")
        print(f"\n  PR:")
        print(f"    {scenario['pr'][:120]}...")
        print(f"\n  Review ({len(mems)} memories retrieved):")
        for line in review.strip().splitlines():
            print(f"    {line}")

        # Cleanup so reruns start fresh
        memory.delete_user(agent_id)

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
