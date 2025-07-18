CLASS ltc_prime_length DEFINITION FINAL
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_hello FOR TESTING,
      test_abcdcba FOR TESTING,
      test_kittens FOR TESTING,
      test_orange FOR TESTING,
      test_wow FOR TESTING,
      test_world FOR TESTING,
      test_madam_mixed_case FOR TESTING,
      test_wow_mixed_case FOR TESTING,
      test_empty_string FOR TESTING,
      test_hi FOR TESTING,
      test_go FOR TESTING,
      test_gogo FOR TESTING,
      test_long_string FOR TESTING,
      test_madam_proper_case FOR TESTING,
      test_single_char_m FOR TESTING,
      test_single_char_digit FOR TESTING.
ENDCLASS.

CLASS ltc_prime_length IMPLEMENTATION.

  METHOD test_hello.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'Hello' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_abcdcba.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'abcdcba' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_kittens.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'kittens' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_orange.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'orange' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_wow.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'wow' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_world.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'world' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_madam_mixed_case.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'MadaM' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_wow_mixed_case.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'Wow' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_empty_string.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( '' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_hi.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'HI' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_go.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'go' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_gogo.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'gogo' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_long_string.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'aaaaaaaaaaaaaaa' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_madam_proper_case.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'Madam' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_single_char_m.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( 'M' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_single_char_digit.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_082=>prime_length( '1' )
      exp = abap_false ).
  ENDMETHOD.

ENDCLASS.