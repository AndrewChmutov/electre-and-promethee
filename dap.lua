local dap = require("dap")

dap.configurations.python = {
    {
        type = "python",
        request = "launch",
        name = "Promethee",
        module = "promethee.main",
        console = "integratedTerminal",
        args = { "data" },
    },
    {
        type = "python",
        request = "launch",
        name = "Electre",
        module = "electre_tri_b.main",
        console = "integratedTerminal",
        args = { "data" },
    },
}
