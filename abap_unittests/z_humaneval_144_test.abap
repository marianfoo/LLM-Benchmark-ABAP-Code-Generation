CLASS ltcl_test_simplify DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test1 FOR TESTING,
      test2 FOR TESTING,
      test3 FOR TESTING,
      test4 FOR TESTING,
      test5 FOR TESTING,
      test6 FOR TESTING,
      test7 FOR TESTING,
      test8 FOR TESTING,
      test9 FOR TESTING,
      test10 FOR TESTING,
      test11 FOR TESTING,
      test12 FOR TESTING,
      test13 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_simplify IMPLEMENTATION.

  METHOD test1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_144=>simplify( x = '1/5' n = '5/1' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_144=>simplify( x = '1/6' n = '2/1' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test3.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_144=>simplify( x = '5/1' n = '3/1' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_144=>simplify( x = '7/10' n = '10/2' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test5.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_144=>simplify( x = '2/10' n = '50/10' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test6.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_144=>simplify( x = '7/2' n = '4/2' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test7.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_144=>simplify( x = '11/6' n = '6/1' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test8.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_144=>simplify( x = '2/3' n = '5/2' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test9.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_144=>simplify( x = '5/2' n = '3/5' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test10.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_144=>simplify( x = '2/4' n = '8/4' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test11.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_144=>simplify( x = '2/4' n = '4/2' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test12.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_144=>simplify( x = '1/5' n = '5/1' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test13.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_144=>simplify( x = '1/5' n = '1/5' )
      exp = abap_false ).
  ENDMETHOD.

ENDCLASS.