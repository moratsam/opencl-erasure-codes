package cmd

import (
	"strconv"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/moratsam/opencl-erasure-codes/codec"
	cl "github.com/moratsam/opencl-erasure-codes/pu/opencl"
	vl "github.com/moratsam/opencl-erasure-codes/pu/vanilla"
)


var (
	k int
	n int
	file_in	string
	file_out string
	proc		string	
	c codec.Codec

	root_cmd = &cobra.Command{
		Use:		"erco",
		Short:	"Code data with polynomial erasure codes.",
		Long:		`The only thing that matters is the silent endurance of the few,
						whose impassible presence as "stone guests" helps to create
						new relationships, new distances, new values, and helps to
						construct a pole that, although it will certainly not prevent
						this world inhabited by the distracted and restless from
						being what it is, will still help to transmit to someone the
						sensation of the truth - a sensation that could become for
						them the principle of a liberating crisis.`,
		PersistentPreRun:	func(_ *cobra.Command, _ []string) {
			c = getCodec()
		},

	}

	cmd_codec = &cobra.Command{
		Use: "codec",
		Run: func(cmd *cobra.Command, args []string) {
			err := c.Encode(byte(k), byte(n), file_in)	
			check(err)
			shards := make([]string, n)
			for i := range shards {
				shards[i] = file_in + "_" + strconv.Itoa(i) + ".enc"
			}
			err = c.Decode(shards, file_out)
			check(err)
		},
	}

	cmd_decode = &cobra.Command{
		Use: "decode",
		Run: func(cmd *cobra.Command, args []string) {
			shards := viper.GetStringSlice("shards")
			err := c.Decode(shards, file_out)
			check(err)
		},
	}

	cmd_encode = &cobra.Command{
		Use: "encode",
		Run: func(cmd *cobra.Command, args []string) {
			err := c.Encode(byte(k), byte(n), file_in)	
			check(err)
		},
	}
)

func getCodec() codec.Codec {
	switch proc{
	case "standalone":
		pu, err := cl.NewOpenCLPU(n)
		check(err)
		return codec.NewCodec(pu)
	case "streamer":
		pu, err := cl.NewStreamerPU()
		check(err)
		return codec.NewStreamerCodec(pu)
	case "vanilla":
		pu := vl.NewVanillaPU()
		return codec.NewCodec(pu)
	default:
		panic("wrong processor selection")
	}
	return nil
}

func Execute() error {
	iit()
	return root_cmd.Execute()
}

func iit() {
	root_cmd.AddCommand(cmd_codec, cmd_decode, cmd_encode)

	// Cmd Root
	root_cmd.PersistentFlags().StringVarP(&proc, "proc", "p", "standalone", "Choose processor type ({\"standalone\",\"streamer\",\"vanilla\"}, default \"standalone\")")

	// Cmd Codec
	cmd_codec.Flags().IntVarP(&k, "k", "", 6, "Number of redundant shards to be created")
	cmd_codec.Flags().IntVarP(&n, "n", "", 7, "Number of shards needed to reconstruct original data")
	cmd_codec.Flags().StringVarP(&file_in, "input", "i", "", "Input file")
	cmd_codec.Flags().StringVarP(&file_out, "output", "o", "", "Output file")
	cmd_codec.MarkFlagRequired("input")
	cmd_codec.MarkFlagRequired("output")

	// Cmd Decode
	cmd_decode.Flags().StringVarP(&file_out, "output", "o", "", "Output file")
	cmd_decode.Flags().StringSlice("shards", []string{}, "List of shard file names")
	viper.BindPFlag("shards", cmd_decode.Flags().Lookup("shards"))
	cmd_decode.MarkFlagRequired("input")
	cmd_decode.MarkFlagRequired("shards")

	// Cmd Encode
	cmd_encode.Flags().IntVarP(&k, "k", "", 6, "Number of redundant shards to be created")
	cmd_encode.Flags().IntVarP(&n, "n", "", 7, "Number of shards needed to reconstruct original data")
	cmd_encode.Flags().StringVarP(&file_in, "input", "i", "", "Input file")
	cmd_encode.MarkFlagRequired("input")
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
