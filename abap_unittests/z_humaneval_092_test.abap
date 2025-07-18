CLASS z_humaneval_092_test DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING.
ENDCLASS.

CLASS z_humaneval_092_test IMPLEMENTATION.

  METHOD test_case_1.
    cl_abap_unit_assert=>assert_true(
      act = z_humaneval_092=>any_int( x = 2 y = 3 z = 1 )
      msg = 'This prints if this assert fails 1 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_2.
    cl_abap_unit_assert=>assert_false(
      act = z_humaneval_092=>any_int( x = '2.5' y = 2 z = 3 )
      msg = 'This prints if this assert fails 2 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_3.
    cl_abap_unit_assert=>assert_false(
      act = z_humaneval_092=>any_int( x = '1.5' y = 5 z = '3.5' )
      msg = 'This prints if this assert fails 3 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_4.
    cl_abap_unit_assert=>assert_false(
      act = z_humaneval_092=>any_int( x = 2 y = 6 z = 2 )
      msg = 'This prints if this assert fails 4 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_5.
    cl_abap_unit_assert=>assert_true(
      act = z_humaneval_092=>any_int( x = 4 y = 2 z = 2 )
      msg = 'This prints if this assert fails 5 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_6.
    cl_abap_unit_assert=>assert_false(
      act = z_humaneval_092=>any_int( x = '2.2' y = '2.2' z = '2.2' )
      msg = 'This prints if this assert fails 6 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_7.
    cl_abap_unit_assert=>assert_true(
      act = z_humaneval_092=>any_int( x = -4 y = 6 z = 2 )
      msg = 'This prints if this assert fails 7 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_8.
    cl_abap_unit_assert=>assert_true(
      act = z_humaneval_092=>any_int( x = 2 y = 1 z = 1 )
      msg = 'This prints if this assert fails 8 (also good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_9.
    cl_abap_unit_assert=>assert_true(
      act = z_humaneval_092=>any_int( x = 3 y = 4 z = 7 )
      msg = 'This prints if this assert fails 9 (also good for debugging!)' ).
  ENDMETHOD.

ENDCLASS.