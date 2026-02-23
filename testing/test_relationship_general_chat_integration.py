from chorus_engine.models.relationship import Relationship, RelationshipSurface


def test_standard_conversation_auto_binds_relationship_when_missing(client, db):
    created = client.post(
        "/conversations",
        json={"character_id": "test_char", "title": "Auto-bound relationship", "source": "web"},
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["relationship_id"]

    rel_rows = (
        db.query(Relationship)
        .filter(
            Relationship.owner_user_id == "user:local:owner",
            Relationship.character_id == "test_char",
        )
        .all()
    )
    assert len(rel_rows) == 1
    assert rel_rows[0].id == body["relationship_id"]

    surface_rows = (
        db.query(RelationshipSurface)
        .filter(
            RelationshipSurface.relationship_id == body["relationship_id"],
            RelationshipSurface.surface_id == "web",
            RelationshipSurface.surface_instance_id == "",
        )
        .all()
    )
    assert len(surface_rows) == 1


def test_general_chat_resolver_is_idempotent_and_separated_from_standard_list(client, db):
    first = client.post("/characters/test_char/general-chat")
    assert first.status_code == 200, first.text
    first_body = first.json()

    second = client.post("/characters/test_char/general-chat")
    assert second.status_code == 200, second.text
    second_body = second.json()

    assert second_body["conversation"]["id"] == first_body["conversation"]["id"]
    assert second_body["thread"]["id"] == first_body["thread"]["id"]
    assert second_body["relationship_id"] == first_body["relationship_id"]
    assert first_body["conversation"]["conversation_kind"] == "general_chat"
    assert first_body["conversation"]["relationship_id"]
    assert first_body["conversation"]["title"] == "General Chat with Test Character"

    rel_rows = (
        db.query(Relationship)
        .filter(
            Relationship.owner_user_id == "user:local:owner",
            Relationship.character_id == "test_char",
        )
        .all()
    )
    assert len(rel_rows) == 1

    surface_rows = (
        db.query(RelationshipSurface)
        .filter(
            RelationshipSurface.relationship_id == rel_rows[0].id,
            RelationshipSurface.surface_id == "web",
            RelationshipSurface.surface_instance_id == "",
        )
        .all()
    )
    assert len(surface_rows) == 1
    assert surface_rows[0].general_conversation_id == first_body["conversation"]["id"]

    default_list = client.get("/conversations?character_id=test_char")
    assert default_list.status_code == 200, default_list.text
    default_ids = {row["id"] for row in default_list.json()}
    assert first_body["conversation"]["id"] not in default_ids

    created = client.post(
        "/conversations",
        json={"character_id": "test_char", "title": "Standard v0 test", "source": "web"},
    )
    assert created.status_code == 200, created.text
    created_body = created.json()
    standard_id = created_body["id"]
    assert created_body["relationship_id"] == first_body["relationship_id"]

    default_list_after = client.get("/conversations?character_id=test_char")
    assert default_list_after.status_code == 200, default_list_after.text
    default_ids_after = {row["id"] for row in default_list_after.json()}
    assert standard_id in default_ids_after
    assert first_body["conversation"]["id"] not in default_ids_after

    gc_only = client.get("/conversations?character_id=test_char&conversation_kind=general_chat")
    assert gc_only.status_code == 200, gc_only.text
    gc_ids = {row["id"] for row in gc_only.json()}
    assert first_body["conversation"]["id"] in gc_ids
    assert standard_id not in gc_ids
