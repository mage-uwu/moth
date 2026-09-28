# Generates TinyStories-style RPG dialogue with an open-weights teacher (Qwen3-4B-Instruct-2507, Apache 2.0)
# on a GPU with vLLM: sequence-level distillation for a tiny student. Each prompt combines a random setting,
# pair of characters, mood, situation, twist and three required simple words (the TinyStories recipe for
# diversity), and asks for simple words in one fixed format, the same one data/prep_rpg.py writes:
#   ~ <place>
#   <one sentence about the place>
#   <speaker>: <line>
# Scenes that break the format are dropped. Output is appended in chunks, so a partial file is usable.
# usage (GPU): pip install vllm; python3 data/gen_synth.py --out rpg_synth.txt --mb 20
#        (CPU):  python3 data/gen_synth.py --dry 3        prints three prompts
import argparse, random, re, sys, time

SETTINGS = """a busy tavern|a quiet chapel|the castle gate|a blacksmith's forge|a dark forest path|a mountain pass|
a fishing village dock|a wizard's tower|a royal throne room|a market square|a dragon's cave|a haunted crypt|
a farm at harvest|a river ferry|a desert oasis|a snowy watchtower|a goblin camp|an elf grove|a dwarf mine|
a pirate ship|a ruined temple|a roadside inn|a prison cell|a library of old books|an alchemist's shop|
a field of flowers|a bridge guarded by a troll|a swamp hut|a city wall at night|a royal garden|a stable|
a bakery at dawn|a graveyard|a lighthouse|a mushroom cave|a camp by a fire|a hidden valley|a treasure vault|
a frozen lake|a battlefield after the fight|a tower of bells|a fairy ring|a hunting lodge|a monastery kitchen|
a crossroads shrine|a sunken ship|a giant's table|a jail wagon|an arena|a tailor's shop""".replace("\n", "").split("|")
CHARACTERS = """knight|squire|old wizard|young witch|blacksmith|innkeeper|thief|guard|king|queen|princess|prince|
farmer|fisher|merchant|bard|priest|nun|goblin|troll|dragon|elf archer|dwarf miner|ghost|pirate captain|
healer|hunter|shepherd|baker|beggar|ranger|orc warrior|talking cat|talking crow|giant|fairy|alchemist|
stable boy|kitchen maid|court jester|old soldier|mapmaker|monk|necromancer|paladin|tailor|ferryman|
lighthouse keeper|librarian|gravedigger|spy|herbalist|village child|traveling cook|bounty hunter|
retired adventurer|lost apprentice|frog prince|skeleton|mermaid""".replace("\n", "").split("|")
MOODS = "friendly|tense|funny|sad|mysterious|hopeful|angry|scared|proud|curious|sleepy|excited|grumpy|kind|nervous".split("|")
SITUATIONS = """one asks the other for help with a quest|they argue over who owns a strange item|one wants to buy something
but has little gold|a stranger asks for directions|one warns the other about danger ahead|they plan how to cross a
river|one has lost something important|they meet for the first time|one wants to join the other's journey|they try to
open a locked door|one needs a cure for a sick friend|they share news from a far town|one is hiding a secret|they
make a deal|one tries to cheer up the other|they guard something through the night|one teaches the other a skill|
they find a map|one asks about an old legend|they must choose which road to take|one has broken a rule|they prepare
for a feast|one wants revenge but the other talks them out of it|they hear a strange noise|one returns after a long
time away""".replace("\n", " ").split("|")
TWISTS = """a riddle|a joke|a lie that is found out|a surprise at the end|a promise|a warning|a small kindness|
a funny mistake|a bargain|an apology|a clever trick|a brave choice|a gift|a secret name|a song""".replace("\n", "").split("|")
WORDS = """apple bread candle sword shield coin key map rope boat horse cloak crown ring potion stone moon star
river bridge tower door window fire smoke rain snow wind storm cave forest flower seed honey milk cheese soup
pot spoon lamp bell drum flute song dance dream shadow ghost bone egg feather nest wolf bear fox owl frog fish
cat dog mouse rat spider bee goat sheep cow pig chicken dragon tooth claw tail wing gold silver iron glass
paper letter book ink secret promise gift friend enemy king queen child mother father brother sister
laugh cry shout whisper sleep wake hide seek find lose run climb jump swim fly fall build break fix open
close carry follow wait hurry brave quiet angry happy tired hungry cold warm dark bright heavy tiny huge
old young strange lucky clever silly kind cruel true false sweet bitter""".split()

PROMPT = """Write a short scene of spoken dialogue from a fantasy role-playing game.

Use exactly this format and nothing else:
~ <name of the place>
<one short sentence describing the place>
<speaker>: <what they say>
<speaker>: <what they say>

Rules:
- The speakers are {a} and {b}. Write each speaker's name in lowercase, exactly as given, before every line.
- Write 6 to 12 lines of dialogue. Only spoken words: no narration, no actions in brackets, no titles, no markdown.
- Use simple words that a young child would understand. Keep sentences short.
- Every line must follow from the one before it. The characters stay true to who they are.
- Place: {place}. Mood: {mood}. What happens: {situation}. The scene must include {twist}.
- Use these words somewhere in the dialogue: {w1}, {w2}, {w3}."""

LINE = re.compile(r"^[a-z][a-z' -]{0,40}: \S")

def make_prompt(rng):
    a, b = rng.sample(CHARACTERS, 2)
    w = rng.sample(WORDS, 3)
    return PROMPT.format(a=a, b=b, place=rng.choice(SETTINGS), mood=rng.choice(MOODS), situation=rng.choice(SITUATIONS),
                         twist=rng.choice(TWISTS), w1=w[0], w2=w[1], w3=w[2]), (a, b)

def clean(text, speakers):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip() and not l.strip().startswith("```")]
    if len(lines) < 8 or not lines[0].startswith("~ ") or LINE.match(lines[1]): return None
    talk = lines[2:]
    if not all(LINE.match(l) and l.split(":", 1)[0] in speakers for l in talk): return None
    return "\n".join([" ".join(l.split()) for l in lines])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rpg_synth.txt"); ap.add_argument("--mb", type=float, default=20)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=2048); ap.add_argument("--dry", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if args.dry:
        for _ in range(args.dry): print(make_prompt(rng)[0], end="\n\n")
        sys.exit()
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.92, seed=args.seed)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=600, seed=args.seed)
    target, written, kept, tried, toks, t0 = int(args.mb * 1e6), 0, 0, 0, 0, time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        while written < target:
            batch = [make_prompt(rng) for _ in range(args.chunk)]
            chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p, _ in batch]
            for (_, spk), o in zip(batch, llm.generate(chats, sp, use_tqdm=False)):
                tried += 1; toks += len(o.outputs[0].token_ids)
                s = clean(o.outputs[0].text, spk)
                if s is None: continue
                s = ("\n\n" if written else "") + s
                f.write(s); written += len(s.encode()); kept += 1
            f.flush()
            dt = time.time() - t0
            print(f"{written / 1e6:.1f} MB | {kept}/{tried} scenes kept | {toks / dt:.0f} tok/s | {dt / 60:.1f} min", flush=True)
        f.write("\n")
    print("DONE", flush=True)
