CLASS lhc_test DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS check_simple FOR TESTING.
    METHODS check_complex FOR TESTING.

ENDCLASS.

CLASS lhc_test IMPLEMENTATION.

  METHOD check_simple.
    DATA(result) = z_humaneval_035=>max_element( iv_list = VALUE int4_table( ( 1 ) ( 2 ) ( 3 ) ) ).
    ASSERT result = 3.
  ENDMETHOD.

  METHOD check_complex.
    DATA(result) = z_humaneval_035=>max_element( iv_list = VALUE int4_table(
      ( 5 ) ( 3 ) ( -5 ) ( 2 ) ( -3 ) ( 3 ) ( 9 ) ( 0 ) ( 124 ) ( 1 ) ( -10 ) ) ).
    ASSERT result = 124.
  ENDMETHOD.

ENDCLASS.