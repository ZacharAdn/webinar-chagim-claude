#!/usr/bin/env bash
# Put the loop back to where the ladder left it: no verdicts, rules v1 active,
# outcome columns empty. For rehearsing the loop more than once on one project.
# The records and the logged predictions stay. Needs the same psql and password
# file as supabase_setup.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="$(python3 -c 'import tomllib;print(tomllib.load(open("ladder.toml","rb"))["project"]["name"])')"
REGION="$(python3 -c 'import tomllib;print(tomllib.load(open("ladder.toml","rb"))["project"].get("region","eu-central-1"))')"
REF="$(python3 -c 'import tomllib,re;u=tomllib.load(open(".streamlit/secrets.toml","rb"))["connections"]["supabase"]["SUPABASE_URL"];print(re.search(r"//([^.]+)\.",u).group(1))')"
PSQL="${PSQL:-/opt/homebrew/opt/libpq/bin/psql}"
export PGPASSWORD="$(cat "$HOME/.supabase/$NAME.dbpass")"

for H in "aws-1-$REGION.pooler.supabase.com" "aws-0-$REGION.pooler.supabase.com"; do
  DB="host=$H port=5432 user=postgres.$REF dbname=postgres sslmode=require"
  "$PSQL" "$DB" -tA -c 'select 1' >/dev/null 2>&1 && break
done

"$PSQL" "$DB" -v ON_ERROR_STOP=1 -q <<'SQL'
delete from public.feedback;
delete from public.rules where version > 1;
update public.rules set active = (version = 1);
update public.predictions set outcome = null, outcome_at = null;
SQL
echo "loop reset: 0 verdicts, rules v1 active"
