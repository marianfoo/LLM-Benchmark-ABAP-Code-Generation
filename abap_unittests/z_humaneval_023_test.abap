CLASS lt_humaneval_023 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS test_strlen_empty FOR TESTING.
    METHODS test_strlen_single_char FOR TESTING.
    METHODS test_strlen_multiple_chars FOR TESTING.
ENDCLASS.

CLASS lt_humaneval_023 IMPLEMENTATION.

  METHOD test_strlen_empty.
    DATA(length) = z_humaneval_023=>strlen( '' ).
    cl_abap_unit_assert=>assert_equals( act = length exp = 0 ).
  ENDMETHOD.

  METHOD test_strlen_single_char.
    DATA(length) = z_humaneval_023=>strlen( 'x' ).
    cl_abap_unit_assert=>assert_equals( act = length exp = 1 ).
  ENDMETHOD.

  METHOD test_strlen_multiple_chars.
    DATA(length) = z_humaneval_023=>strlen( 'asdasnakj' ).
    cl_abap_unit_assert=>assert_equals( act = length exp = 9 ).
  ENDMETHOD.

ENDCLASS.