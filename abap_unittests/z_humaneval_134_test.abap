CLASS ltcl_test_check_last_char DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_apple FOR TESTING,
      test_apple_pi_e FOR TESTING,
      test_eeeee FOR TESTING,
      test_single_a FOR TESTING,
      test_pumpkin_pie_space FOR TESTING,
      test_pumpkin_pie_1 FOR TESTING,
      test_empty_string FOR TESTING,
      test_eeeee_e_space FOR TESTING,
      test_apple_pie FOR TESTING,
      test_apple_pi_e_space FOR TESTING.

ENDCLASS.

CLASS ltcl_test_check_last_char IMPLEMENTATION.

  METHOD test_apple.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_134=>check_if_last_char_is_a_letter( 'apple' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_apple_pi_e.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_134=>check_if_last_char_is_a_letter( 'apple pi e' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_eeeee.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_134=>check_if_last_char_is_a_letter( 'eeeee' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_single_a.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_134=>check_if_last_char_is_a_letter( 'A' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_pumpkin_pie_space.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_134=>check_if_last_char_is_a_letter( |Pumpkin pie | )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_pumpkin_pie_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_134=>check_if_last_char_is_a_letter( 'Pumpkin pie 1' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_empty_string.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_134=>check_if_last_char_is_a_letter( '' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_eeeee_e_space.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_134=>check_if_last_char_is_a_letter( |eeeee e | )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_apple_pie.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_134=>check_if_last_char_is_a_letter( 'apple pie' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_apple_pi_e_space.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_134=>check_if_last_char_is_a_letter( |apple pi e | )
      exp = abap_false ).
  ENDMETHOD.

ENDCLASS.