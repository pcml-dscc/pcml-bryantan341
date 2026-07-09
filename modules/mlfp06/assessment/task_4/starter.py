def solve() -> dict:
    """Compile the org, attach envelopes, run verify_action, test escalation.

    See problem.md for the exact envelope specs and the return contract:
        {"org_stats": {...}, "verdicts": [bool x10], "escalation_caught": bool}
    """

    # TODO 1: Compile the organisation with compile_governance() -> (engine, org).
    #         Read org.n_agents / org.n_delegations / org.n_departments.

    engine, org = compile_governance()

    org_stats = {
        "n_agents": org.n_agents,
        "n_delegations": org.n_delegations,
        "n_departments": org.n_departments,
    }

    # TODO 2: Build a ConstraintEnvelopeConfig for each of the four roles
    #         (all 5 dimensions populated). Attach each to the engine via
    #         engine.set_role_envelope(RoleEnvelope(...)) using the addresses
    #         above. Budgets + allowed actions are in problem.md.

    role_specs = {
        "data_analyst": {
            "delegator": "chief_ml_officer",
            "clearance": ConfidentialityLevel.RESTRICTED,
            "budget": 20.0,
            "actions": [
                "read_data",
                "summarise_data",
                "generate_report",
            ],
        },
        "model_trainer": {
            "delegator": "chief_ml_officer",
            "clearance": ConfidentialityLevel.RESTRICTED,
            "budget": 100.0,
            "actions": [
                "train_model",
                "evaluate_model",
                "read_data",
            ],
        },
        "risk_assessor": {
            "delegator": "chief_risk_officer",
            "clearance": ConfidentialityLevel.RESTRICTED,
            "budget": 200.0,
            "actions": [
                "read_data",
                "audit_model",
                "generate_report",
                "access_audit_log",
            ],
        },
        "customer_agent": {
            "delegator": "vp_customer",
            "clearance": ConfidentialityLevel.PUBLIC,
            "budget": 5.0,
            "actions": [
                "answer_question",
                "search_faq",
            ],
        },
    }

    for role, spec in role_specs.items():

        envelope = ConstraintEnvelopeConfig(
            id=f"{role}_envelope",
            description=f"{role} least privilege envelope",
            confidentiality_clearance=spec["clearance"],
            financial=FinancialConstraintConfig(
                max_spend_usd=spec["budget"],
            ),
            operational=OperationalConstraintConfig(
                allowed_actions=spec["actions"],
            ),
            temporal=TemporalConstraintConfig(),
            data_access=DataAccessConstraintConfig(),
            communication=CommunicationConstraintConfig(),
        )

        engine.set_role_envelope(
            RoleEnvelope(
                role_address=AGENT_ADDRESSES[role],
                delegator_address=DELEGATOR_ADDRESSES[spec["delegator"]],
                envelope=envelope,
            )
        )

    # TODO 3: For each case in CASES (in order), call
    #         engine.verify_action(role_address=..., action=..., context={"cost": ...})
    #         and append verdict.allowed (a bool) to a verdicts list.

    verdicts = []

    for role, action, cost in CASES:
        verdict = engine.verify_action(
            role_address=AGENT_ADDRESSES[role],
            action=action,
            context={"cost": cost},
        )
        verdicts.append(verdict.allowed)

    # TODO 4: Build a CONFIDENTIAL parent envelope for vp_customer and a rogue
    #         RESTRICTED child that escalates budget + actions. Call
    #         RoleEnvelope.validate_tightening(parent_envelope=..., child_envelope=...)
    #         inside try/except MonotonicTighteningError; set escalation_caught.

    parent_envelope = ConstraintEnvelopeConfig(
        id="parent",
        description="VP Customer Parent",
        confidentiality_clearance=ConfidentialityLevel.CONFIDENTIAL,
        financial=FinancialConstraintConfig(
            max_spend_usd=50.0,
        ),
        operational=OperationalConstraintConfig(
            allowed_actions=[
                "answer_question",
                "search_faq",
            ],
        ),
        temporal=TemporalConstraintConfig(),
        data_access=DataAccessConstraintConfig(),
        communication=CommunicationConstraintConfig(),
    )

    rogue_child = ConstraintEnvelopeConfig(
        id="rogue",
        description="Escalation Attempt",
        confidentiality_clearance=ConfidentialityLevel.RESTRICTED,
        financial=FinancialConstraintConfig(
            max_spend_usd=1000.0,
        ),
        operational=OperationalConstraintConfig(
            allowed_actions=[
                "answer_question",
                "search_faq",
                "read_data",
                "deploy_model",
            ],
        ),
        temporal=TemporalConstraintConfig(),
        data_access=DataAccessConstraintConfig(),
        communication=CommunicationConstraintConfig(),
    )

    escalation_caught = False

    try:
        RoleEnvelope.validate_tightening(
            parent_envelope=parent_envelope,
            child_envelope=rogue_child,
        )
    except MonotonicTighteningError:
        escalation_caught = True

    return {
        "org_stats": org_stats,
        "verdicts": verdicts,
        "escalation_caught": escalation_caught,
    }