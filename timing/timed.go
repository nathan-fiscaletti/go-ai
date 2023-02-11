package timing

import "time"

func Timed(c func()) time.Duration {
	start := time.Now()
	c()
	return time.Since(start)
}
