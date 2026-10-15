export const SAMPLES: { label: string; query: string; context: string }[] = [
  {
    label: "Support / refund policy",
    query: "What is the enterprise refund policy?",
    context: `The enterprise refund policy allows a full refund within 30 days of purchase, provided the licence has not been activated.

General consumer refunds are handled differently and require proof of purchase.

Refund requests may be submitted through the billing portal at any time.

The enterprise refund policy requires approval from the account manager if the licence has been activated.

Refunds for annual plans are prorated for the unused period.

The marketing team publishes a weekly newsletter about product updates.`,
  },
  {
    label: "Legal clause lookup",
    query: "What is the termination notice period for the supplier agreement?",
    context: `The supplier agreement may be terminated by either party with 60 days written notice.

The customer agreement requires 30 days notice for termination.

Either party may terminate the supplier agreement immediately in the event of insolvency.

The supplier agreement contains an auto-renewal clause of twelve months.

Notice periods for the supplier agreement must be delivered in writing by registered post.`,
  },
  {
    label: "Clean context (low risk)",
    query: "What is the refund window?",
    context: `Refunds are available within 30 days of purchase.

Customers must provide proof of purchase to request a refund.`,
  },
];