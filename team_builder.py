"""
Pokemon Team Builder 
==============================================================

Each LLM receives the same competitive format rules and must produce 
a team of 6 Pokemon in structured JSON (validated with Pydantic).
A tool queries PokeAPI to provide the LLM with real data (stats, types,
abilities) instead of letting it make up numbers.


Setup:
    pip install datapizza-ai datapizza-ai-clients-openai datapizza-ai-clients-anthropic requests

    create your .env file with your API keys:
        OPENAI_API_KEY=...
        ANTHROPIC_API_KEY=...

Usage:
    python team_builder.py
"""

import json
import os
from typing import Literal

import requests
from pydantic import BaseModel, Field

from datapizza.agents import Agent
from datapizza.tools import tool

from dotenv import load_dotenv
load_dotenv()

POKEAPI_BASE = "https://pokeapi.co/api/v2"


# ---------------------------------------------------------------------------
# 1. Output scheme: force the LLM to respond with a valid structure
# ---------------------------------------------------------------------------

class MoveSet(BaseModel):
    moves: list[str] = Field(..., min_length=4, max_length=4)


class TeamMember(BaseModel):
    species: str
    item: str
    ability: str
    nature: str
    moves: list[str] = Field(..., min_length=4, max_length=4)
    ev_hp: int = Field(..., ge=0, le=252)
    ev_atk: int = Field(..., ge=0, le=252)
    ev_def: int = Field(..., ge=0, le=252)
    ev_spa: int = Field(..., ge=0, le=252)
    ev_spd: int = Field(..., ge=0, le=252)
    ev_spe: int = Field(..., ge=0, le=252)
    tera_type: str | None = None


class Team(BaseModel):
    format: str
    members: list[TeamMember] = Field(..., min_length=6, max_length=6)
    strategy_notes: str = Field(description="2-3 sentences on the general strategy of the team")


# ---------------------------------------------------------------------------
# 2. Tool: real data from PokeAPI (stats, types, abilities) to avoid hallucinations
# ---------------------------------------------------------------------------


@tool(description="Retrieve types, base stats, and available abilities for a Pokemon from PokeAPI.")
def get_pokemon_data(species_name: str) -> str:
    name = species_name.lower().strip().replace(" ", "-")
    try:
        resp = requests.get(f"{POKEAPI_BASE}/pokemon/{name}", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Error retrieving data for '{species_name}': {e}"

    data = resp.json()
    stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}
    types = [t["type"]["name"] for t in data["types"]]
    abilities = [a["ability"]["name"] for a in data["abilities"]]

    return json.dumps(
        {"name": data["name"], "types": types, "base_stats": stats, "abilities": abilities},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 3. Base legality validator (does not replace Showdown, but filters
#    the most obvious errors before even starting a battle)
# ---------------------------------------------------------------------------

def validate_team(team: Team) -> list[str]:
    errors = []
    species_seen = set()
    allowed_stats = {"hp", "atk", "def", "spa", "spd", "spe"}

    for member in team.members:
        if member.species.lower() in species_seen:
            errors.append(f"Species clause violated: {member.species} duplicated.")
        species_seen.add(member.species.lower())

        if len(member.moves) != 4 or len(set(member.moves)) != 4:
            errors.append(f"{member.species}: must have exactly 4 distinct moves.")

        total_evs = member.ev_hp + member.ev_atk + member.ev_def + member.ev_spa + member.ev_spd + member.ev_spe
        if total_evs > 510:
            errors.append(f"{member.species}: total EVs {total_evs} > 510.")
        for stat_name, val in [("hp", member.ev_hp), ("atk", member.ev_atk), ("def", member.ev_def),
                                ("spa", member.ev_spa), ("spd", member.ev_spd), ("spe", member.ev_spe)]:
            if val > 252 or val < 0:
                errors.append(f"{member.species}: EV {stat_name}={val} fuori range 0-252.")

    return errors


# ---------------------------------------------------------------------------
# 4. Client builders: each provider/model combination gets its own Agent
# ---------------------------------------------------------------------------

def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def call_client(provider, model):
    if provider == "openai":
        from datapizza.clients.openai import OpenAIClient
        return OpenAIClient(api_key=os.environ["OPENAI_API_KEY"], model=model)
    elif provider == "anthropic":
        from datapizza.clients.anthropic import AnthropicClient
        return AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"], model=model)
    else:
        raise ValueError(f"Provider not supported: {provider}")       

def build_research_agent(client, provider, model) -> Agent:
    """Agent with tool, without output_cls — collects candidate Pokémon data."""

    return Agent(
        name=f"{provider}-{model}",
        client=client,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        tools=[get_pokemon_data],
        max_steps=12,
        gen_args={"max_tokens": 4096}
    )

def build_compiler_agent(client, provider, model) -> Agent:

    return Agent(
        name=f"{provider}-{model}",
        client=client,
        system_prompt=COMPILER_SYSTEM_PROMPT,
        output_cls=Team,
        max_steps=12,
        gen_args={"max_tokens": 4096}
    )

def run_pipeline(provider, model) -> Team:
    client = call_client(provider, model)
    research_agent = build_research_agent(client, provider, model)
    research_task = f"Gather data on candidates for a {FORMAT} team"
    research_result = research_agent.run(research_task)
    with open(f"outputs/{provider}-{model}/research_output.txt", "w", encoding="utf-8") as f:
        f.write(research_result.text)

    compiler_agent = build_compiler_agent(client, provider, model)
    compile_task = f"Collected data:\n\n{research_result.text}\n\nBuild the final team.\n\nRespond exclusively with the requested object."
    compile_result = compiler_agent.run(compile_task)

    return compile_result.structured_data[0]  # istanza di Team


# ---------------------------------------------------------------------------
# 5. System prompt: instructions for the LLM to build a competitive team
# ---------------------------------------------------------------------------
FORMAT = 'VGC'
rules = load_text(f"rules/{FORMAT}.txt")

RESEARCH_SYSTEM_PROMPT = f"""You are a competitive Pokemon player experienced in the {FORMAT} format.

Format rules:
{rules}

Your task in this step is ONLY to research candidate Pokemon for a team —
you are NOT building the final team yet.

Use the get_pokemon_data tool to check real stats, types, and abilities
before considering a Pokemon: do not make up numbers from memory.

Explore candidates covering different roles (physical sweeper, special sweeper,
wall, hazard setter, support, revenge killer) that fit together as a coherent
team and cover common meta threats. Aim for 8-10 candidates so the next step
has options to choose from.

Report your findings in a clear, organized way: for each candidate, note its
role, key stats, typing, and why it fits the team. Do not produce a final
JSON team here, just the research. Output only what is requested, no extra text or commentary."""


COMPILER_SYSTEM_PROMPT = f"""You are a competitive Pokemon player experienced in the {FORMAT} format.

Format rules:
{rules}

You will receive research notes on candidate Pokemon collected in a previous
step. Your task is to select exactly 6 of them and compile the final team,
with full movesets, EVs, items, natures, and abilities, respecting the
constraints below.

Provide a competitively sensible EV spread to every Pokemon. Never leave EVs empty.

Constraints:
- Species clause: no duplicate Pokemon.
- Each Pokemon has exactly 4 distinct moves.
- Total EVs per Pokemon <= 510, individual EV <= 252.
- Adhere to the banlist of the indicated format.

Respond exclusively with the requested structured object."""

COMPETITORS = [
    #("openai", "gpt-4.1"),
    ("anthropic", "claude-sonnet-4-6"),
    # add other providers/models as needed
]

def main():
    results = {}
    for provider, model in COMPETITORS:
        os.makedirs(f"outputs/{provider}-{model}", exist_ok=True)
        print(f"\n=== {provider} / {model} ===")
        team = run_pipeline(provider, model)
        print(team.model_dump_json(indent=2))

        errors = validate_team(team)

        results[f"{provider}-{model}"] = {
            "team": team.model_dump(),
            "validation_errors": errors,
        }

        print(f"Team: {[m.species for m in team.members]}")
        print("Legal" if not errors else f"Errors: {errors}")

        with open(f"outputs/{provider}-{model}/team.json", "w", encoding="utf-8") as f:
            json.dump(team, f, ensure_ascii=False, indent=2)

    with open(f"outputs/{provider}-{model}/team.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nTeams saved to teams.json")

if __name__ == "__main__":
    main()