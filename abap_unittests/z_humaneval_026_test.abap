CLASS z_humaneval_026_demo DEFINITION
  FINAL
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS test_remove_duplicates_empty FOR TESTING.
    METHODS test_remove_duplicates_none FOR TESTING.
    METHODS test_remove_duplicates_some FOR TESTING.
ENDCLASS.

CLASS z_humaneval_026_demo IMPLEMENTATION.

  METHOD test_remove_duplicates_empty.
    DATA(result) = z_humaneval_026=>remove_duplicates( VALUE int4_table( ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ) ).
  ENDMETHOD.

  METHOD test_remove_duplicates_none.
    DATA(result) = z_humaneval_026=>remove_duplicates(
      VALUE int4_table( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ) ).
  ENDMETHOD.

  METHOD test_remove_duplicates_some.
    DATA(result) = z_humaneval_026=>remove_duplicates(
      VALUE int4_table( ( 1 ) ( 2 ) ( 3 ) ( 2 ) ( 4 ) ( 3 ) ( 5 ) ) ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 1 ) ( 4 ) ( 5 ) ) ).
  ENDMETHOD.

ENDCLASS.