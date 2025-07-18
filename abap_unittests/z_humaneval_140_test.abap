CLASS ltcl_test_fix_spaces DEFINITION
FOR TESTING
DURATION SHORT
RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_simple_case FOR TESTING,
      test_trailing_space FOR TESTING,
      test_double_spaces FOR TESTING,
      test_triple_spaces FOR TESTING,
      test_leading_spaces FOR TESTING.

ENDCLASS.

CLASS ltcl_test_fix_spaces IMPLEMENTATION.

  METHOD test_simple_case.
    DATA: result TYPE string.
    result = z_humaneval_140=>fix_spaces( 'Example' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 'Example'
      msg = 'This prints if this assert fails 1 (good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_trailing_space.
    DATA: result TYPE string.
    result = z_humaneval_140=>fix_spaces( |Mudasir Hanif | ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 'Mudasir_Hanif_'
      msg = 'This prints if this assert fails 2 (good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_double_spaces.
    DATA: result TYPE string.
    result = z_humaneval_140=>fix_spaces( 'Yellow Yellow  Dirty  Fellow' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 'Yellow_Yellow__Dirty__Fellow'
      msg = 'This prints if this assert fails 3 (good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_triple_spaces.
    DATA: result TYPE string.
    result = z_humaneval_140=>fix_spaces( 'Exa   mple' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 'Exa-mple'
      msg = 'This prints if this assert fails 4 (good for debugging!)'
    ).
  ENDMETHOD.

  METHOD test_leading_spaces.
    DATA: result TYPE string.
    result = z_humaneval_140=>fix_spaces( '   Exa 1 2 2 mple' ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = '-Exa_1_2_2_mple'
      msg = 'This prints if this assert fails 4 (good for debugging!)'
    ).
  ENDMETHOD.

ENDCLASS.