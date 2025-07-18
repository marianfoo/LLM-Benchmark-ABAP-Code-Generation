CLASS ltcl_test_z_humaneval_143 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_this_is_a_test FOR TESTING,
      test_lets_go_for_swimming FOR TESTING,
      test_there_is_no_place FOR TESTING,
      test_hi_i_am_hussein FOR TESTING,
      test_go_for_it FOR TESTING,
      test_here FOR TESTING,
      test_here_is FOR TESTING.
ENDCLASS.

CLASS ltcl_test_z_humaneval_143 IMPLEMENTATION.
  METHOD test_this_is_a_test.
    DATA: result TYPE string.
    result = z_humaneval_143=>words_in_sentence( 'This is a test' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 'is' ).
  ENDMETHOD.

  METHOD test_lets_go_for_swimming.
    DATA: result TYPE string.
    result = z_humaneval_143=>words_in_sentence( 'lets go for swimming' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 'go for' ).
  ENDMETHOD.

  METHOD test_there_is_no_place.
    DATA: result TYPE string.
    result = z_humaneval_143=>words_in_sentence( 'there is no place available here' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 'there is no place' ).
  ENDMETHOD.

  METHOD test_hi_i_am_hussein.
    DATA: result TYPE string.
    result = z_humaneval_143=>words_in_sentence( 'Hi I am Hussein' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 'Hi am Hussein' ).
  ENDMETHOD.

  METHOD test_go_for_it.
    DATA: result TYPE string.
    result = z_humaneval_143=>words_in_sentence( 'go for it' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 'go for it' ).
  ENDMETHOD.

  METHOD test_here.
    DATA: result TYPE string.
    result = z_humaneval_143=>words_in_sentence( 'here' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = '' ).
  ENDMETHOD.

  METHOD test_here_is.
    DATA: result TYPE string.
    result = z_humaneval_143=>words_in_sentence( 'here is' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 'is' ).
  ENDMETHOD.
ENDCLASS.