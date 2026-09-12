#!/usr/bin/env python3

import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO
# ============================================================

COMMANDS_DIR = Path(
    "/home/gustavo/Documentos/Multi-model-Federated-Learning/examples/TETC"
)


# ============================================================
# HIPERPARÂMETROS
# ============================================================

# Strategy EXATA que deve ser executada.
#
# Exemplo no TXT:
#     --strategy='MultiFedAvg+MFP_v2'
#
STRATEGY = "MultiFedAvg+MFP_v2"


# Tipos de drift que devem ser executados.
#
# Pode indicar um ou vários:
#
# DRIFT_TYPES = ["sudden"]
# DRIFT_TYPES = ["gradual"]
# DRIFT_TYPES = ["sudden", "gradual"]
#
DRIFT_TYPES = [
    "sudden",
    "gradual",
]


# Tipos de shift que devem ser executados.
#
# Pode indicar um ou vários:
#
# SHIFT_TYPES = ["concept"]
# SHIFT_TYPES = ["label"]
# SHIFT_TYPES = ["concept", "label"]
#
SHIFT_TYPES = [
    # "concept",
    "label",
]


# ============================================================
# FORMATO DOS ARQUIVOS TXT
# ============================================================
#
# Os arquivos TXT possuem comandos neste formato:
#
# python main.py ... --strategy='MultiFedAvg+MFP_v2'
#     ... --experiment_id='label_shift#0.1-1.0_gradual'
#
# ou:
#
# python main.py ... --strategy='MultiFedAvg+MFP_v2'
#     ... --experiment_id='concept_drift#0.1-1.0_gradual'
#
# Portanto:
#
#   label_shift#...  -> SHIFT_TYPE = "label"
#   concept_drift#... -> SHIFT_TYPE = "concept"
#
# E o último componente do experiment_id indica:
#
#   ..._sudden  -> DRIFT_TYPE = "sudden"
#   ..._gradual -> DRIFT_TYPE = "gradual"
#
# Não existe necessariamente um argumento explícito
# --shift ou --drift_type nos comandos.
#
# ============================================================


# ============================================================
# FUNÇÕES PARA EXTRAIR ARGUMENTOS
# ============================================================

def get_argument(command, argument_names):
    """
    Obtém o valor de um argumento do comando.

    Suporta:

        --argument=value
        --argument='value'
        --argument="value"
        --argument value
        --argument 'value'
        --argument "value"
    """

    for name in argument_names:

        # ----------------------------------------------------
        # --argument='value'
        # --argument="value"
        # ----------------------------------------------------

        pattern = (
            rf"{re.escape(name)}\s*=\s*"
            rf"(['\"])(.*?)\1"
        )

        match = re.search(pattern, command)

        if match:
            return match.group(2).strip()

        # ----------------------------------------------------
        # --argument=value
        # ----------------------------------------------------

        pattern = (
            rf"{re.escape(name)}\s*=\s*([^\s]+)"
        )

        match = re.search(pattern, command)

        if match:
            return match.group(1).strip("\"'")

        # ----------------------------------------------------
        # --argument 'value'
        # --argument "value"
        # ----------------------------------------------------

        pattern = (
            rf"{re.escape(name)}\s+"
            rf"(['\"])(.*?)\1"
        )

        match = re.search(pattern, command)

        if match:
            return match.group(2).strip()

        # ----------------------------------------------------
        # --argument value
        # ----------------------------------------------------

        pattern = (
            rf"{re.escape(name)}\s+([^\s]+)"
        )

        match = re.search(pattern, command)

        if match:
            return match.group(1).strip("\"'")

    return None


# ============================================================
# EXTRAÇÃO DA STRATEGY
# ============================================================

def get_strategy(command):
    """
    Extrai somente o argumento --strategy.

    A comparação posterior é EXATA.

    Exemplo:

        --strategy='MultiFedAvg+MFP_v2'

    resulta em:

        MultiFedAvg+MFP_v2
    """

    return get_argument(
        command,
        [
            "--strategy",
        ],
    )


# ============================================================
# EXTRAÇÃO DO EXPERIMENT_ID
# ============================================================

def get_experiment_id(command):
    """
    Extrai o experiment_id do comando.

    Exemplo:

        --experiment_id='label_shift#0.1-1.0_gradual'

    retorna:

        label_shift#0.1-1.0_gradual
    """

    return get_argument(
        command,
        [
            "--experiment_id",
            "--experiment-id",
        ],
    )


# ============================================================
# EXTRAÇÃO DO SHIFT TYPE
# ============================================================

def get_shift_type(command):
    """
    Identifica o tipo de shift a partir do experiment_id.

    Exemplos:

        label_shift#0.1-1.0_gradual
            -> label

        concept_drift#0.1-1.0_gradual
            -> concept

    Também reconhece combined, caso exista nos arquivos:

        combined_shift#...
        combined_drift#...
    """

    experiment_id = get_experiment_id(command)

    if experiment_id is None:
        return None

    experiment_id_lower = experiment_id.lower()

    # --------------------------------------------------------
    # Label shift
    # --------------------------------------------------------

    if (
        experiment_id_lower.startswith("label_shift#")
        or experiment_id_lower.startswith("label_shift_")
    ):
        return "label"

    # --------------------------------------------------------
    # Concept drift / concept shift
    # --------------------------------------------------------

    if (
        experiment_id_lower.startswith("concept_drift#")
        or experiment_id_lower.startswith("concept_drift_")
        or experiment_id_lower.startswith("concept_shift#")
        or experiment_id_lower.startswith("concept_shift_")
    ):
        return "concept"

    # --------------------------------------------------------
    # Combined shift/drift
    # --------------------------------------------------------

    if (
        experiment_id_lower.startswith("combined_shift#")
        or experiment_id_lower.startswith("combined_shift_")
        or experiment_id_lower.startswith("combined_drift#")
        or experiment_id_lower.startswith("combined_drift_")
    ):
        return "combined"

    return None


# ============================================================
# EXTRAÇÃO DO DRIFT TYPE
# ============================================================

def get_drift_type(command):
    """
    Identifica sudden ou gradual a partir do final do
    experiment_id.

    Exemplos:

        label_shift#0.1-1.0_gradual
            -> gradual

        label_shift#0.1-1.0_sudden
            -> sudden

        concept_drift#1.0-10.0_gradual
            -> gradual
    """

    experiment_id = get_experiment_id(command)

    if experiment_id is None:
        return None

    experiment_id_lower = experiment_id.lower()

    # O tipo de drift é o último campo depois do "_".
    last_part = experiment_id_lower.rsplit("_", 1)[-1]

    if last_part in ("sudden", "gradual"):
        return last_part

    return None


# ============================================================
# FILTRO
# ============================================================

def command_matches(command):
    """
    O comando é selecionado somente quando:

        strategy == STRATEGY
        AND
        drift_type in DRIFT_TYPES
        AND
        shift_type in SHIFT_TYPES
    """

    command_strategy = get_strategy(command)
    command_drift = get_drift_type(command)
    command_shift = get_shift_type(command)

    # --------------------------------------------------------
    # Strategy
    # --------------------------------------------------------

    if command_strategy != STRATEGY:
        return False

    # --------------------------------------------------------
    # Drift
    # --------------------------------------------------------

    if command_drift is None:
        return False

    if not any(
        command_drift.lower() == drift.lower()
        for drift in DRIFT_TYPES
    ):
        return False

    # --------------------------------------------------------
    # Shift
    # --------------------------------------------------------

    if command_shift is None:
        return False

    if not any(
        command_shift.lower() == shift.lower()
        for shift in SHIFT_TYPES
    ):
        return False

    return True


# ============================================================
# LEITURA DOS ARQUIVOS TXT
# ============================================================

def extract_commands():
    """
    Lê todos os arquivos .txt da pasta.

    Cada linha que começa com 'python ' ou 'python3 '
    é considerada um comando.

    Cabeçalhos como:

        Global
        Local
        0.1-1.0_gradual

    são ignorados automaticamente.

    A ordem dos comandos é preservada:
        1. arquivos em ordem alfabética;
        2. linhas na ordem em que aparecem.
    """

    txt_files = sorted(
        COMMANDS_DIR.glob("*.txt")
    )

    if not txt_files:

        print(
            f"\nERRO: nenhum arquivo TXT encontrado em:\n"
            f"{COMMANDS_DIR}"
        )

        sys.exit(1)

    commands = []

    print("\nArquivos TXT encontrados:")

    for txt_file in txt_files:
        print(f"  - {txt_file.name}")

    # --------------------------------------------------------
    # Lê cada arquivo
    # --------------------------------------------------------

    for txt_file in txt_files:

        with txt_file.open(
            "r",
            encoding="utf-8"
        ) as f:

            lines = f.readlines()

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            line = line.strip()

            # ------------------------------------------------
            # Ignora linhas vazias
            # ------------------------------------------------

            if not line:
                continue

            # ------------------------------------------------
            # Ignora comentários
            # ------------------------------------------------

            if line.startswith("#"):
                continue

            # ------------------------------------------------
            # Extrai somente comandos Python
            # ------------------------------------------------

            if (
                line.startswith("python ")
                or line.startswith("python3 ")
                or line.startswith("python\t")
                or line.startswith("python3\t")
            ):

                commands.append(
                    {
                        "command": line,
                        "file": txt_file.name,
                        "line": line_number,
                    }
                )

    return commands


# ============================================================
# EXECUÇÃO DE UM COMANDO
# ============================================================

def execute_command(index, total, item):

    command = item["command"]

    print("\n")
    print("=" * 100)
    print(
        f"EXECUTANDO COMANDO {index}/{total}"
    )
    print(
        f"Arquivo: {item['file']}"
    )
    print(
        f"Linha:   {item['line']}"
    )
    print("-" * 100)

    # Mostra explicitamente o comando atual.
    print("COMANDO ATUAL:")
    print(command)

    print("=" * 100)

    start_time = time.time()

    try:

        # shlex.split remove corretamente as aspas do comando,
        # incluindo:
        #
        # --strategy='MultiFedAvg+MFP_v2'
        #
        args = shlex.split(command)

        # subprocess.run bloqueia até o processo terminar.
        # Portanto, os comandos são executados sequencialmente.
        result = subprocess.run(args)

    except Exception as e:

        print("\nERRO AO EXECUTAR O COMANDO:")
        print(e)

        return False

    elapsed = (
        time.time()
        - start_time
    )

    if result.returncode != 0:

        print("\n✗ COMANDO FALHOU")

        print(
            f"Código de retorno: "
            f"{result.returncode}"
        )

        print(
            f"Tempo: "
            f"{elapsed / 60:.2f} minutos"
        )

        return False

    print(
        "\n✓ Comando concluído com sucesso."
    )

    print(
        f"Tempo: "
        f"{elapsed / 60:.2f} minutos"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("#" * 100)
    print("EXECUTOR SEQUENCIAL DE EXPERIMENTOS")
    print("#" * 100)

    print(
        f"\nDiretório dos comandos:"
        f"\n{COMMANDS_DIR}"
    )

    print(
        f"\nStrategy EXATA:"
        f"\n{STRATEGY}"
    )

    print(
        "\nDrift types selecionados:"
    )

    for drift in DRIFT_TYPES:
        print(f"  - {drift}")

    print(
        "\nShift types selecionados:"
    )

    for shift in SHIFT_TYPES:
        print(f"  - {shift}")

    # --------------------------------------------------------
    # Lê os comandos
    # --------------------------------------------------------

    all_commands = extract_commands()

    print(
        f"\nTotal de comandos Python encontrados: "
        f"{len(all_commands)}"
    )

    # --------------------------------------------------------
    # Filtra
    # --------------------------------------------------------

    selected_commands = [
        item
        for item in all_commands
        if command_matches(item["command"])
    ]

    # --------------------------------------------------------
    # Mostra seleção
    # --------------------------------------------------------

    print(
        f"\nTotal de comandos selecionados: "
        f"{len(selected_commands)}"
    )

    if not selected_commands:

        print(
            "\nNenhum comando corresponde à "
            "configuração selecionada."
        )

        print(
            f"\nSTRATEGY = {STRATEGY}"
        )

        print(
            f"DRIFT_TYPES = {DRIFT_TYPES}"
        )

        print(
            f"SHIFT_TYPES = {SHIFT_TYPES}"
        )

        sys.exit(1)

    print("\n" + "-" * 100)
    print("COMANDOS SELECIONADOS:")
    print("-" * 100)

    for i, item in enumerate(
        selected_commands,
        start=1
    ):

        command = item["command"]

        print(
            f"\n[{i}] "
            f"{item['file']}:{item['line']}"
        )

        print(
            f"    Strategy: "
            f"{get_strategy(command)}"
        )

        print(
            f"    Shift:    "
            f"{get_shift_type(command)}"
        )

        print(
            f"    Drift:    "
            f"{get_drift_type(command)}"
        )

        print(
            f"    Comando:"
        )

        print(
            f"    {command}"
        )

    # --------------------------------------------------------
    # Confirmação
    # --------------------------------------------------------

    print("\n" + "-" * 100)

    response = input(
        "\nExecutar os comandos selecionados? [s/N]: "
    ).strip().lower()

    if response not in (
        "s",
        "sim",
        "y",
        "yes",
    ):

        print(
            "\nExecução cancelada."
        )

        return

    # --------------------------------------------------------
    # Execução sequencial
    # --------------------------------------------------------

    total = len(selected_commands)

    global_start = time.time()

    for i, item in enumerate(
        selected_commands,
        start=1
    ):

        success = execute_command(
            i,
            total,
            item,
        )

        # ----------------------------------------------------
        # Interrompe se houver erro.
        # ----------------------------------------------------

        if not success:

            print("\n")
            print("#" * 100)
            print("EXECUÇÃO INTERROMPIDA")
            print("#" * 100)

            print(
                f"\nComando que falhou: "
                f"{i}/{total}"
            )

            print(
                "\nOs comandos seguintes "
                "não serão executados."
            )

            sys.exit(1)

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    total_time = (
        time.time()
        - global_start
    )

    print("\n")
    print("#" * 100)
    print("TODOS OS COMANDOS FORAM CONCLUÍDOS")
    print("#" * 100)

    print(
        f"\nComandos executados: "
        f"{total}"
    )

    print(
        f"Tempo total: "
        f"{total_time / 3600:.2f} horas"
    )


# ============================================================
# EXECUÇÃO DO SCRIPT
# ============================================================

if __name__ == "__main__":
    main()